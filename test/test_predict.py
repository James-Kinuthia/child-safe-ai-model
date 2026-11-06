import unittest
from unittest.mock import patch

from ml.nlp import predict


class PredictMessageTests(unittest.TestCase):
    def test_includes_context_with_current_message(self):
        with patch.object(predict.model, "predict", return_value=["grooming"]) as model_predict:
            result = predict.predict_message("current message", "incoming: previous message")

        self.assertEqual(result, "grooming")
        model_predict.assert_called_once_with(
            ["Conversation context:\nincoming: previous message\n\nCurrent message:\ncurrent message"]
        )

    def test_preserves_single_message_input_without_context(self):
        with patch.object(predict.model, "predict", return_value=["safe"]) as model_predict:
            result = predict.predict_message("single message")

        self.assertEqual(result, "safe")
        model_predict.assert_called_once_with(["single message"])


if __name__ == "__main__":
    unittest.main()
