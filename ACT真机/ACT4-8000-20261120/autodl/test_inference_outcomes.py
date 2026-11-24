"""Offline outcome accounting regressions using temporary event files only."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import inference_outcomes as outcomes


class InferenceOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="outcome-test-", dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.events = Path(self.temporary.name) / "trial_events.jsonl"
        self.records = []

    def write_events(self):
        self.events.write_text("".join(json.dumps(record) + "\n" for record in self.records), encoding="utf-8")

    def trial(self, trial_id, reason="time_limit_unknown", ended=True, outcome="unknown"):
        self.records.append({"event": "trial_start", "trial_id": trial_id, "episode_id": f"episode-{trial_id}"})
        if ended:
            self.records.append({"event": "trial_end", "trial_id": trial_id, "reason": reason, "outcome": outcome})
        self.write_events()

    def test_timer_finish_and_ongoing_trials_remain_unknown(self):
        self.trial(1)
        self.trial(2, reason="operator_finish")
        self.trial(3, ended=False)
        result = outcomes.summarize(self.events)
        self.assertEqual((result["attempted"], result["completed"], result["unknown"]), (3, 2, 3))
        self.assertIsNone(result["success_rate"])
        self.assertEqual(result["success_fraction_lower_bound"], 0)
        self.assertEqual(result["success_fraction_upper_bound"], 1)
        self.assertFalse(outcomes.labels_path(self.events).exists())

    def test_f8_is_failed_and_all_definitive_failures_have_zero_rate(self):
        self.trial(1, reason="operator_failed", outcome="failed")
        result = outcomes.summarize(self.events)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["unknown"], 0)
        self.assertEqual(result["success_rate"], 0)
        self.assertEqual(result["success_rate_denominator"], 1)

    def test_unrecognised_automatic_success_does_not_fabricate_success(self):
        self.trial(1, outcome="success")
        result = outcomes.summarize(self.events)
        self.assertEqual(result["success"], 0)
        self.assertEqual(result["unknown"], 1)

    def test_manual_label_and_f8_failure_produce_complete_success_rate(self):
        self.trial(1, reason="operator_failed", outcome="failed")
        self.trial(2)
        source_before = self.events.read_bytes()
        outcomes.label(self.events, 2, "success", note="Reviewed video, cube lifted and placed.")
        result = outcomes.summarize(self.events)
        self.assertEqual((result["success"], result["failed"], result["unknown"]), (1, 1, 0))
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["manual_labelled"], 1)
        self.assertEqual(self.events.read_bytes(), source_before)

    def test_partial_label_rate_has_explicit_smaller_denominator(self):
        self.trial(1)
        self.trial(2)
        outcomes.label(self.events, 1, "success")
        result = outcomes.summarize(self.events)
        self.assertIsNone(result["success_rate"])
        self.assertEqual(result["labelled_success_rate"], 1)
        self.assertEqual(result["labelled_rate_denominator"], 1)
        self.assertEqual(result["success_fraction_lower_bound"], 0.5)
        self.assertEqual(result["success_fraction_upper_bound"], 1)

    def test_duplicate_manual_label_requires_correction_and_preserves_history(self):
        self.trial(1)
        first = outcomes.label(self.events, 1, "success")
        label_file = outcomes.labels_path(self.events)
        prior = label_file.read_bytes()
        with self.assertRaisesRegex(ValueError, "--correct"):
            outcomes.label(self.events, 1, "failed")
        self.assertEqual(label_file.read_bytes(), prior)
        correction = outcomes.label(self.events, 1, "failed", correct=True, note="Cube slipped before placement.")
        self.assertEqual(correction["supersedes_label_id"], first["label_id"])
        self.assertTrue(label_file.read_bytes().startswith(prior))
        self.assertEqual(outcomes.summarize(self.events)["failed"], 1)
        self.assertEqual(len(label_file.read_text(encoding="utf-8").splitlines()), 2)

    def test_overriding_f8_requires_explicit_correction(self):
        self.trial(1, reason="operator_failed", outcome="failed")
        with self.assertRaisesRegex(ValueError, "--correct"):
            outcomes.label(self.events, 1, "success")
        correction = outcomes.label(self.events, 1, "success", correct=True)
        self.assertEqual(correction["supersedes_automatic_outcome"], "failed")
        self.assertEqual(outcomes.summarize(self.events)["success_rate"], 1)

    def test_unseen_or_unfinished_trial_cannot_be_labelled(self):
        self.trial(1, ended=False)
        for trial_id in (1, 2):
            with self.assertRaisesRegex(ValueError, "both trial_start and trial_end"):
                outcomes.label(self.events, trial_id, "success")

    def test_missing_earlier_trial_is_counted_as_unknown(self):
        self.trial(2, reason="operator_failed", outcome="failed")
        result = outcomes.summarize(self.events)
        self.assertEqual((result["attempted"], result["completed"], result["unknown"]), (2, 1, 1))
        self.assertIsNone(result["success_rate"])
        self.assertEqual(result["missing_trial_start"], [1])

    def test_label_from_other_cloud_episode_is_rejected(self):
        self.trial(1)
        outcomes.label(self.events, 1, "success")
        self.records[0]["episode_id"] = "different-session"
        self.write_events()
        with self.assertRaisesRegex(ValueError, "another cloud episode"):
            outcomes.summarize(self.events)

    def test_duplicate_trial_event_and_corrupt_input_fail_explicitly(self):
        self.trial(1)
        self.records.append(self.records[0])
        self.write_events()
        with self.assertRaisesRegex(ValueError, "Duplicate trial_start"):
            outcomes.summarize(self.events)
        self.events.write_text('{"event":', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            outcomes.summarize(self.events)

    def test_input_size_is_bounded(self):
        self.trial(1)
        with mock.patch.object(outcomes, "MAX_EVENT_BYTES", 8):
            with self.assertRaisesRegex(ValueError, "size/record limit"):
                outcomes.summarize(self.events)

    def test_concurrent_label_writer_is_rejected(self):
        self.trial(1)
        with outcomes.locked_labels(outcomes.labels_path(self.events), create=True):
            with self.assertRaisesRegex(RuntimeError, "labels are busy"):
                outcomes.label(self.events, 1, "success")

    def test_missing_final_newline_cannot_corrupt_prior_labels(self):
        self.trial(1)
        outcomes.label(self.events, 1, "success")
        path = outcomes.labels_path(self.events)
        prior = path.read_bytes().rstrip(b"\n")
        path.write_bytes(prior)
        with self.assertRaisesRegex(ValueError, "incomplete final line"):
            outcomes.label(self.events, 1, "failed", correct=True)
        self.assertEqual(path.read_bytes(), prior)

    def test_cli_label_and_summary(self):
        self.trial(1)
        with redirect_stdout(io.StringIO()) as output:
            status = outcomes.main(["label", "--events", str(self.events), "--trial-id", "1",
                                    "--outcome", "success", "--note", "Reviewed"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["outcome"], "success")
        with redirect_stdout(io.StringIO()) as output:
            status = outcomes.main(["summarize", "--events", str(self.events)])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["success_rate"], 1)


if __name__ == "__main__":
    unittest.main()
