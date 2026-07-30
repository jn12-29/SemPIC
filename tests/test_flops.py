import unittest

from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from sempic.cache_comb.utils.flops import AutoFlopsCalculator


class FlopsCalculatorTests(unittest.TestCase):
    def test_llama_prefill_boundaries(self):
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        ))
        calculator = AutoFlopsCalculator(model)

        layer_flops = 16_520
        decoder_flops = 49_560
        final_norm_flops = 320
        lm_head_flops = 1_024

        self.assertEqual(calculator.decoder_layer_flops(2, 5, 7), layer_flops)
        self.assertEqual(calculator.total_flops(2, 5, 7), decoder_flops)
        self.assertEqual(
            calculator.body_flops(2, 5, 7),
            decoder_flops + final_norm_flops,
        )
        self.assertEqual(
            calculator.forward_flops(2, 5, 7),
            decoder_flops + final_norm_flops + lm_head_flops,
        )
        self.assertEqual(calculator.output_flops(2), 64 + lm_head_flops)
        self.assertEqual(
            calculator.output_flops(2, hidden_rows=3, logits_rows=2),
            192 + 2_048,
        )

    def test_qwen3_prefill_boundaries(self):
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        ))
        calculator = AutoFlopsCalculator(model)

        layer_flops = 17_000
        decoder_flops = 51_000
        final_norm_flops = 320
        lm_head_flops = 1_024

        self.assertEqual(calculator.decoder_layer_flops(2, 5, 7), layer_flops)
        self.assertEqual(calculator.total_flops(2, 5, 7), decoder_flops)
        self.assertEqual(
            calculator.body_flops(2, 5, 7),
            decoder_flops + final_norm_flops,
        )
        self.assertEqual(
            calculator.forward_flops(2, 5, 7),
            decoder_flops + final_norm_flops + lm_head_flops,
        )
        self.assertEqual(calculator.output_flops(2), 64 + lm_head_flops)
        self.assertEqual(
            calculator.forward_flops(2, 5, 7, logits_rows=2),
            decoder_flops + final_norm_flops + 2_048,
        )


if __name__ == "__main__":
    unittest.main()
