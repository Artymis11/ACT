"""Tony's actual input helpers adapted to normalized Windows XInput frames.

Only six hash-pinned AST function definitions are compiled. The upstream module
itself is never imported or executed: no pygame, rebot controller, SDK, CAN,
upstream constructor, run, shutdown, zero, or enable path is reachable here.

The output is a seven-axis normalized INTENT, not a command or feedback. This
module does not enforce LB/B, freshness, pause, or session admission; its caller
must apply those guards before turning any intent into a planned movement.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
UPSTREAM_REPOSITORY="https://github.com/TonyWeyyy/reBot_B601_rs_xbox_teleop"
UPSTREAM_COMMIT="f402f80ccab68d457178b5a8c688642147e069a4"
UPSTREAM_SOURCE_SHA256="9d33cebed3598f396d325e255399602353e33a1d5f909b9e1d242c0a0736d5eb"
UPSTREAM_MANIFEST_SHA256="c687ddce353c0ecd4346127bfa5e9f021c40f1a800703fa870572322cbed0860"
SOURCE_FUNCTIONS=("apply_curve","clamp","XboxTeleop._ctl","XboxTeleop._read_axis",
                  "XboxTeleop._read_trigger","XboxTeleop._read_button")
AXES=("LX","LY","RX","RY","LT","RT")
BUTTON_NAMES=("A","B","X","Y","LB","RB","Back","Start","LeftStick","RightStick",
              "DPadUp","DPadDown","DPadLeft","DPadRight")
BUTTON_MASKS=(0x1000,0x2000,0x4000,0x8000,0x0100,0x0200,0x0020,0x0010,
              0x0040,0x0080,0x0001,0x0002,0x0004,0x0008)
CONTROL_NAMES=frozenset(("lb","rb","pause","resume","return_origin","lx","ly","rx","ry",
                         "lt","rt","up","down","left","right"))


class TonyInputError(ValueError):
    pass


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def verify_upstream_snapshot(root=ROOT):
    """Verify the frozen manifest and every original byte without imports."""
    root=Path(root).resolve()
    manifest_path=root/"tony_source_manifest.json"
    try:
        raw=manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=UPSTREAM_MANIFEST_SHA256:
            raise TonyInputError("Tony source manifest differs from the pinned snapshot")
        manifest=json.loads(raw)
        if (manifest.get("repository")!=UPSTREAM_REPOSITORY or manifest.get("commit")!=UPSTREAM_COMMIT
                or manifest.get("source_file_count")!=14 or len(manifest.get("files",[]))!=14):
            raise TonyInputError("Tony snapshot identity or file count differs")
        paths=set()
        for row in manifest["files"]:
            relative=Path(row["path"])
            path=(root/relative).resolve()
            if (relative.is_absolute() or ".." in relative.parts
                    or not path.is_relative_to(root/"vendor"/"tony_xbox_teleop")
                    or row["path"] in paths):
                raise TonyInputError("Tony snapshot path is outside its source tree or duplicated")
            paths.add(row["path"])
            data=path.read_bytes()
            if len(data)!=row["bytes"] or hashlib.sha256(data).hexdigest()!=row["sha256"]:
                raise TonyInputError("Tony source bytes changed: "+row["path"])
        source=root/"vendor"/"tony_xbox_teleop"/"xbox_teleop.py"
        if hashlib.sha256(source.read_bytes()).hexdigest()!=UPSTREAM_SOURCE_SHA256:
            raise TonyInputError("Tony input source differs from the pinned commit checkout")
        return manifest
    except (OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
        raise TonyInputError("Tony source snapshot cannot be verified") from exc


def _load_upstream_input(root,deadzone,cubic):
    verify_upstream_snapshot(root)
    source=Path(root)/"vendor"/"tony_xbox_teleop"/"xbox_teleop.py"
    tree=ast.parse(source.read_text(encoding="utf-8"),filename=str(source))
    constants={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ("DEADZONE","USE_CUBIC_CURVE","JOG_SPEEDS_DEG_S","SPEED_LEVELS","LOOP_HZ"):
                    constants[target.id]=ast.literal_eval(node.value)
    if set(constants)!={"DEADZONE","USE_CUBIC_CURVE","JOG_SPEEDS_DEG_S","SPEED_LEVELS","LOOP_HZ"}:
        raise TonyInputError("pinned upstream input constants missing")
    top={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    classes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=="XboxTeleop"]
    if len(classes)!=1:
        raise TonyInputError("pinned XboxTeleop input class missing")
    methods={node.name:node for node in classes[0].body if isinstance(node,ast.FunctionDef)}
    selected=[copy.deepcopy(top[name]) for name in ("apply_curve","clamp")]
    selected.extend(copy.deepcopy(methods[name]) for name in ("_ctl","_read_axis","_read_trigger","_read_button"))
    # Defining just these functions runs none of the upstream top-level code,
    # defaults, imports, class constructor, event loop, or hardware lifecycle.
    module=ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[]))
    namespace={"__builtins__":{"abs":abs,"bool":bool,"float":float,"tuple":tuple,"max":max,"min":min},
               "DEADZONE":deadzone,"USE_CUBIC_CURVE":cubic}
    exec(compile(module,str(source)+"::pinned-input-helpers","exec"),namespace)
    reader_type=type("PinnedTonyInputReader",(),{name:namespace[name] for name in
        ("_ctl","_read_axis","_read_trigger","_read_button")})
    return reader_type,namespace["apply_curve"],constants


def _validate_mapping(mapping):
    if not isinstance(mapping,dict) or mapping.get("schema")!="tony-beitong-xinput-map-v1":
        raise TonyInputError("explicit local XInput mapping schema required; author Android mapping is not accepted")
    if mapping.get("adapter_space")!="normalized_xinput_frame_v1" or tuple(mapping.get("axis_order",()))!=AXES:
        raise TonyInputError("local mapping must use the documented XInput axis space")
    if tuple(mapping.get("button_order",()))!=BUTTON_NAMES:
        raise TonyInputError("local mapping must use the documented XInput button space")
    signs=mapping.get("native_input_signs")
    if not isinstance(signs,list) or len(signs)!=7 or any(type(s) is not int or s not in (-1,1) for s in signs):
        raise TonyInputError("seven explicit input-only signs required")
    if signs[2]!=1:
        raise TonyInputError("this local profile requires positive RY to mean native J3 increase")
    deadzone=mapping.get("deadzone")
    if not _finite(deadzone) or not 0<deadzone<1 or type(mapping.get("use_cubic_curve")) is not bool:
        raise TonyInputError("valid deadzone and explicit curve selection required")
    controls=mapping.get("controls")
    if not isinstance(controls,dict) or set(controls)!=CONTROL_NAMES:
        raise TonyInputError("all local controls must be explicit; no fallback to author defaults")
    for name,c in controls.items():
        if not isinstance(c,dict) or type(c.get("index")) is not int:
            raise TonyInputError("invalid control index: "+name)
        kind,index=c.get("type"),c["index"]
        if kind=="axis":
            if not 0<=index<len(AXES) or not _finite(c.get("value")) or not -1<=c["value"]<=1:
                raise TonyInputError("invalid axis mapping: "+name)
            rest=c.get("rest",0.)
            if not _finite(rest) or not -1<=rest<=1 or abs(c["value"]-rest)<1e-6:
                raise TonyInputError("degenerate axis mapping: "+name)
        elif kind=="button":
            if not 0<=index<len(BUTTON_NAMES):
                raise TonyInputError("invalid button mapping: "+name)
        elif kind=="hat":
            value=c.get("value")
            if (index!=0 or not isinstance(value,list) or len(value)!=2
                    or any(type(v) is not int or v not in (-1,0,1) for v in value) or value==[0,0]):
                raise TonyInputError("invalid or neutral hat mapping: "+name)
        else:
            raise TonyInputError("unsupported mapping type: "+name)
    ry=controls["ry"]
    if ry.get("type")!="axis" or ry["index"]!=3 or ry.get("value",0)<=0:
        raise TonyInputError("this local profile requires positive RY to mean native J3 increase")


class XInputShim:
    """A pure pygame-shaped reader for an already normalized input Frame."""
    def __init__(self,frame):
        if type(getattr(frame,"connected",None)) is not bool:
            raise TonyInputError("connected must be an explicit boolean")
        self.connected=frame.connected
        if not self.connected:
            self.axes=(0.,)*6
            self.buttons=0
            return
        buttons=getattr(frame,"buttons",None)
        if type(buttons) is not int or not 0<=buttons<=0xFFFF:
            raise TonyInputError("XInput button mask must be uint16")
        values=getattr(frame,"axes",None)
        if not isinstance(values,dict) or any(name not in values for name in AXES):
            raise TonyInputError("six normalized XInput axes required")
        checked=[]
        for index,name in enumerate(AXES):
            value=values[name]
            low=0 if index>=4 else -1
            if not _finite(value) or not low<=value<=1:
                raise TonyInputError("invalid normalized XInput axis: "+name)
            checked.append(float(value))
        self.axes=tuple(checked)
        self.buttons=buttons

    @staticmethod
    def _index(index,size):
        if type(index) is not int or not 0<=index<size:
            raise TonyInputError("invalid normalized reader index")
        return index

    def get_numaxes(self):
        return 6

    def get_numbuttons(self):
        return len(BUTTON_NAMES)

    def get_numhats(self):
        return 1

    def get_axis(self,index):
        return self.axes[self._index(index,6)]

    def get_button(self,index):
        return bool(self.buttons&BUTTON_MASKS[self._index(index,len(BUTTON_NAMES))])

    def get_hat(self,index):
        self._index(index,1)
        up,down=bool(self.buttons&0x0001),bool(self.buttons&0x0002)
        left,right=bool(self.buttons&0x0004),bool(self.buttons&0x0008)
        return (int(right)-int(left),int(up)-int(down))


@dataclass(frozen=True)
class TonyIntent:
    jog: tuple[float,...]
    buttons: dict[str,bool]
    connected: bool


class TonyInputMapper:
    def __init__(self,mapping_path=None,*,root=ROOT):
        self.root=Path(root).resolve()
        path=Path(mapping_path) if mapping_path is not None else self.root/"gamepad_map.beitong_xinput.json"
        try:
            self.mapping=json.loads(path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError) as exc:
            raise TonyInputError("local XInput mapping could not be read") from exc
        _validate_mapping(self.mapping)
        self._reader_type,self._curve,self._upstream_constants=_load_upstream_input(self.root,self.mapping["deadzone"],self.mapping["use_cubic_curve"])

    @property
    def upstream_constants(self):
        return copy.deepcopy(self._upstream_constants)

    @property
    def upstream_metadata(self):
        return {**self.provenance,"constants":self.upstream_constants,
                "effective_curve":{"DEADZONE":self.mapping["deadzone"],"USE_CUBIC_CURVE":self.mapping["use_cubic_curve"]},
                "gripper_conflict_policy":"both raw triggers above deadzone mean zero gripper intent",
                "native_input_signs":list(self.mapping["native_input_signs"])}

    @property
    def provenance(self):
        return {"repository":UPSTREAM_REPOSITORY,"commit":UPSTREAM_COMMIT,
                "source_sha256":UPSTREAM_SOURCE_SHA256,"source_functions":list(SOURCE_FUNCTIONS),
                "execution_scope":"only pinned input helpers; no upstream lifecycle or top-level execution",
                "profile":self.mapping["name"],"input_signs_are_act_mapping":False,"hardware_opened":False}

    def map(self,frame):
        shim=XInputShim(frame)
        if not shim.connected:
            return TonyIntent((0.,)*7,{name:False for name in ("lb","rb","pause","resume","return_origin")},False)
        reader=self._reader_type()
        reader.map=self.mapping
        reader.js=shim
        jog=[self._curve(reader._read_axis("lx")),self._curve(reader._read_axis("ly")),
             self._curve(reader._read_axis("ry")),self._curve(reader._read_axis("rx")),
             float(reader._read_button("up"))-float(reader._read_button("down")),
             float(reader._read_button("left"))-float(reader._read_button("right"))]
        lt,rt=reader._read_trigger("lt"),reader._read_trigger("rt")
        # Deliberate conflict protection: both active triggers mean hold, even
        # with unequal pressure; do not turn ambiguous input into gripper motion.
        grip=0. if lt>self.mapping["deadzone"] and rt>self.mapping["deadzone"] else self._curve(rt)-self._curve(lt)
        jog.append(grip)
        jog=tuple(float(value*sign) for value,sign in zip(jog,self.mapping["native_input_signs"]))
        if not all(math.isfinite(value) and -1<=value<=1 for value in jog):
            raise TonyInputError("upstream helpers produced an invalid normalized intent")
        buttons={name:bool(reader._read_button(name)) for name in ("lb","rb","pause","resume","return_origin")}
        return TonyIntent(jog,buttons,True)

    def normalized_jog(self,frame):
        return self.map(frame).jog
