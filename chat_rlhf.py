import argparse
import re

from transformers import AutoModelForCausalLM

from train_rlhf import ByteTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="llama_rl")
    # generation options
    parser.add_argument("--max_prompt_length", default=8, type=str)
    parser.add_argument("--max_length", default=16, type=str)
    parser.add_argument("--temp", default=0.7, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    args = parser.parse_args()

    tokenizer = ByteTokenizer(padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda")

    help_message = f'Please type a single word in lower case within {args.max_prompt_length - 1} letters at one time. For example, type "hello" and press enter.'
    print(help_message)

    while True:
        try:
            prompt = input("nanoRLHF > ")
        except EOFError:
            break

        if not prompt:
            continue
        if prompt == "stop":
            break

        if not re.match(rf"^[a-z]{{,{args.max_prompt_length - 1}}}$", prompt):
            print(f'Invalid prompt "{prompt}". {help_message}')
            continue

        inputs = tokenizer(prompt + "E", return_tensors="pt")
        input_ids = inputs["input_ids"].cuda(non_blocking=True)
        attention_mask = inputs["attention_mask"].cuda(non_blocking=True)
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=args.max_length,
            use_cache=True,
            do_sample=args.temp > 0,
            temperature=args.temp,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        _, input_len = input_ids.shape
        output = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)
        print(output)

    print("Bye")


if __name__ == "__main__":
    main()
