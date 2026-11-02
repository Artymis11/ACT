"""Real upstream input-helper reuse with synthetic frames only; no hardware."""
import ast
import builtins
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import tony_input as tony

ROOT=Path(__file__).resolve().parent


def frame(buttons=0,connected=True,**axes):
    values=dict.fromkeys(tony.AXES,0.)
    values.update(axes)
    return SimpleNamespace(connected=connected,buttons=buttons,axes=values)


@contextmanager
def local_map_file(change):
    mapping=json.loads((ROOT/"gamepad_map.beitong_xinput.json").read_text(encoding="utf-8"))
    change(mapping)
    with tempfile.TemporaryDirectory(dir=ROOT/"reports") as folder:
        path=Path(folder)/"map.json"
        path.write_text(json.dumps(mapping),encoding="utf-8")
        yield path


class ReusedInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapper=tony.TonyInputMapper()

    def test_actual_vendor_snapshot_identity_and_all_fourteen_file_bytes(self):
        manifest=tony.verify_upstream_snapshot()
        self.assertEqual(manifest["commit"],"f402f80ccab68d457178b5a8c688642147e069a4")
        self.assertEqual(len(manifest["files"]),14)
        source=ROOT/"vendor"/"tony_xbox_teleop"/"xbox_teleop.py"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),tony.UPSTREAM_SOURCE_SHA256)

    def test_only_six_source_helpers_exist_without_upstream_lifecycle(self):
        provenance=self.mapper.provenance
        self.assertEqual(len(provenance["source_functions"]),6)
        self.assertFalse(provenance["hardware_opened"])
        self.assertFalse(provenance["input_signs_are_act_mapping"])
        helper=self.mapper._reader_type
        for name in ("__init__","run","shutdown","start"):
            self.assertNotIn(name,helper.__dict__)
        for name in ("_ctl","_read_axis","_read_button","_read_trigger"):
            self.assertIn("pinned-input-helpers",getattr(helper,name).__code__.co_filename)

    def test_construction_and_mapping_cannot_import_upstream_runtime_or_mutate_path(self):
        forbidden=("pygame","rebot","motorbridge")
        before=list(sys.path)
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if any(name==prefix or name.startswith(prefix+".") for prefix in forbidden):
                raise AssertionError("hardware/lifecycle import attempted: "+name)
            return original(name,*args,**kwargs)
        with patch("builtins.__import__",side_effect=guarded):
            mapper=tony.TonyInputMapper()
            intent=mapper.map(frame(LX=1.,RY=1.,RT=1.))
        self.assertEqual(intent.jog,(1.,0.,1.,0.,0.,0.,1.))
        self.assertEqual(sys.path,before)

    def test_actual_source_constants_are_read_as_literals(self):
        constants=self.mapper.upstream_constants
        self.assertEqual(constants["JOG_SPEEDS_DEG_S"],[30.,20.,20.,30.,40.,60.,60.])
        self.assertEqual(constants["SPEED_LEVELS"],[.25,.5,1.])
        self.assertEqual(constants["DEADZONE"],.12)
        constants["JOG_SPEEDS_DEG_S"][0]=999
        self.assertEqual(self.mapper.upstream_constants["JOG_SPEEDS_DEG_S"][0],30.)

    def test_four_sticks_map_to_correct_joints_without_axis_selection(self):
        expected={"LX":0,"LY":1,"RY":2,"RX":3}
        for axis,index in expected.items():
            for sign in (-1.,1.):
                with self.subTest(axis=axis,sign=sign):
                    jog=self.mapper.normalized_jog(frame(**{axis:sign}))
                    self.assertEqual(jog[index],sign)
                    self.assertEqual(sum(value!=0 for value in jog),1)

    def test_deadzone_and_actual_upstream_cubic_curve(self):
        for value in (0.,.1,.12,-.12):
            self.assertEqual(self.mapper.normalized_jog(frame(LX=value))[0],0.)
        expected=((.56-.12)/(1-.12))**3
        self.assertAlmostEqual(self.mapper.normalized_jog(frame(LX=.56))[0],expected)
        self.assertAlmostEqual(self.mapper.normalized_jog(frame(LX=-.56))[0],-expected)

    def test_j3_upward_input_positive_negative_input_never_reinterpreted_as_reference(self):
        positive=self.mapper.map(frame(RY=1.))
        negative=self.mapper.map(frame(RY=-1.))
        self.assertEqual(positive.jog[2],1.)
        self.assertEqual(negative.jog[2],-1.)
        self.assertNotIn("target",positive.__dict__)
        self.assertNotEqual(positive.jog[2],.093)

    def test_dpad_independent_directions_and_diagonal(self):
        for mask,joint,sign in ((1,4,1.),(2,4,-1.),(4,5,1.),(8,5,-1.)):
            with self.subTest(mask=mask):
                jog=self.mapper.normalized_jog(frame(buttons=mask))
                self.assertEqual(jog[joint],sign)
        jog=self.mapper.normalized_jog(frame(buttons=1|8))
        self.assertEqual(jog[4:6],(1.,-1.))

    def test_contradictory_dpad_directions_cancel_only_their_own_axis(self):
        jog=self.mapper.normalized_jog(frame(buttons=1|2|4))
        self.assertEqual(jog[4:6],(0.,1.))
        jog=self.mapper.normalized_jog(frame(buttons=1|2|4|8))
        self.assertEqual(jog[4:6],(0.,0.))

    def test_triggers_open_close_and_unequal_conflict_hold(self):
        self.assertEqual(self.mapper.normalized_jog(frame(RT=1.))[6],1.)
        self.assertEqual(self.mapper.normalized_jog(frame(LT=1.))[6],-1.)
        self.assertEqual(self.mapper.normalized_jog(frame(LT=.3,RT=1.))[6],0.)
        self.assertEqual(self.mapper.normalized_jog(frame(LT=1.,RT=.3))[6],0.)
        self.assertGreater(self.mapper.normalized_jog(frame(LT=.1,RT=1.))[6],0.)

    def test_all_seven_inputs_can_exist_as_intent_before_caller_gating(self):
        intent=self.mapper.map(frame(buttons=1|4|0x2000,LX=1.,LY=1.,RX=1.,RY=1.,RT=1.))
        self.assertEqual(intent.jog,(1.,)*7)
        self.assertTrue(intent.buttons["pause"])
        self.assertFalse(intent.buttons["lb"])

    def test_beitong_xinput_logical_actions_are_explicit(self):
        mask=0x0100|0x0200|0x2000|0x4000|0x8000
        intent=self.mapper.map(frame(buttons=mask))
        self.assertTrue(all(intent.buttons.values()))
        intent.buttons["pause"]=False
        self.assertTrue(self.mapper.map(frame(buttons=mask)).buttons["pause"])

    def test_disconnected_input_never_replays_old_intent(self):
        self.mapper.map(frame(LX=1.,RT=1.,buttons=0x100))
        missing=SimpleNamespace(connected=False,buttons=object(),axes=None)
        intent=self.mapper.map(missing)
        self.assertFalse(intent.connected)
        self.assertEqual(intent.jog,(0.,)*7)
        self.assertFalse(any(intent.buttons.values()))

    def test_invalid_connected_raw_axes_or_button_channel_rejected(self):
        for axes in ({"LX":float("nan")},{"LY":1.1},{"RY":-1.1},{"LT":-.1},{"RT":True}):
            with self.subTest(axes=axes),self.assertRaises(tony.TonyInputError):
                self.mapper.map(frame(**axes))
        for buttons in (True,-1,65536,"1"):
            with self.subTest(buttons=buttons),self.assertRaises(tony.TonyInputError):
                self.mapper.map(frame(buttons=buttons))
        with self.assertRaises(tony.TonyInputError):
            self.mapper.map(SimpleNamespace(connected=True,buttons=0,axes={}))
        with self.assertRaises(tony.TonyInputError):
            self.mapper.map(SimpleNamespace(connected=1,buttons=0,axes=dict.fromkeys(tony.AXES,0.)))

    def test_pygame_shim_has_bounded_strict_indices_and_hat_conflicts(self):
        shim=tony.XInputShim(frame(buttons=1|2|4|8))
        self.assertEqual(shim.get_hat(0),(0,0))
        for method,args in ((shim.get_axis,[-1]),(shim.get_axis,[6]),(shim.get_axis,[True]),
                            (shim.get_button,[14]),(shim.get_hat,[1])):
            with self.assertRaises(tony.TonyInputError):
                method(*args)

    def test_author_android_map_is_not_accepted_or_automatically_loaded(self):
        author=ROOT/"vendor"/"tony_xbox_teleop"/"gamepad_map.json"
        self.assertIn("HongJingDa",json.loads(author.read_text(encoding="utf-8"))["name"])
        with self.assertRaises(tony.TonyInputError):
            tony.TonyInputMapper(author)
        self.assertIn("Beitong",self.mapper.provenance["profile"])

    def test_malformed_mapping_has_no_author_defaults_fallback(self):
        changes=(lambda m:m["controls"].pop("lb"),
                 lambda m:m["controls"]["pause"].update(index=True),
                 lambda m:m["controls"]["up"].update(type="hat",index=0,value=[0,0]),
                 lambda m:m["controls"]["lt"].update(value=0.,rest=0.))
        for change in changes:
            with local_map_file(change) as path,self.assertRaises(tony.TonyInputError):
                tony.TonyInputMapper(path)

    def test_local_j3_profile_cannot_silently_reverse_native_positive_input(self):
        for change in (lambda m:m["native_input_signs"].__setitem__(2,-1),
                       lambda m:m["controls"]["ry"].update(value=-1),
                       lambda m:m["controls"]["ry"].update(index=2)):
            with local_map_file(change) as path,self.assertRaises(tony.TonyInputError):
                tony.TonyInputMapper(path)

    def test_changed_source_fails_before_compiling_any_payload(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"reports") as folder:
            temp=Path(folder)
            shutil.copyfile(ROOT/"tony_source_manifest.json",temp/"tony_source_manifest.json")
            shutil.copytree(ROOT/"vendor"/"tony_xbox_teleop",temp/"vendor"/"tony_xbox_teleop")
            shutil.copyfile(ROOT/"gamepad_map.beitong_xinput.json",temp/"gamepad_map.beitong_xinput.json")
            source=temp/"vendor"/"tony_xbox_teleop"/"xbox_teleop.py"
            with source.open("ab") as handle:
                handle.write(b"\nraise RuntimeError('untrusted payload must never execute')\n")
            with self.assertRaises(tony.TonyInputError):
                tony.TonyInputMapper(root=temp)

    def test_changed_manifest_is_not_trusted_even_if_its_file_digests_are_updated(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"reports") as folder:
            temp=Path(folder)
            manifest=json.loads((ROOT/"tony_source_manifest.json").read_text(encoding="utf-8"))
            manifest["files"][0]["sha256"]="0"*64
            (temp/"tony_source_manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
            with self.assertRaises(tony.TonyInputError):
                tony.verify_upstream_snapshot(temp)


if __name__=="__main__":
    unittest.main()
