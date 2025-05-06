# Code for: Getting output logits necessary for backprop from the model
#
# Created Date: 9 April 2024
# Last Modified Date: 9 April
# Last Modified Author: Shika

import torch
import os


def get_sentence_logits(model, images, gen_config):
    text = "Describe the quality of the image."

    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    img_embeds = model.encode_img(images)
    text_embeds = model.encode_text(text)
    prompt_embeds = model.wrap_prompt(text_embeds, img_embeds)
    outputs = model.internlm_model.generate(
        inputs_embeds=prompt_embeds, **gen_config)

    # outputs = model.internlm_model.generate(
    #         inputs_embeds=prompt_embs, **gen_config)
    # out_text = model.decode_text(outputs)

    # 5, 78, 4096 as number of beams is set to 5
    first_token_hidden_states = outputs['hidden_states'][0][-1]
    # 5, 1, 4096 as number of beams is set to 5
    final_token_hidden_states = outputs['hidden_states'][-1][-1]

    # Max length of sentence is 500 so len(outputs['hidden_states']) = 500
    # print("Final hidden states shape: ")
    # print(final_token_hidden_states.shape) # 5, 1, 4096

    # just ..... is coming now words
    out_text = model.decode_text(outputs['sequences'])
    # out_text = model.tokenizer.decode(outputs['sequences'][0], skip_special_tokens= True)
    # print(f"The sentence is: {out_text}")

    # exit()
    return final_token_hidden_states

# This get_logits function is from qinstruct code : https://github.com/Q-Future/Q-Instruct/blob/main/eval_scripts/internlm_xcomposer_vl/eval_image_quality.py


def get_init_logits(model, images):
    # report the int() usage because period is before image
    # text = "User: Rate the quality of the image <ImageHere> . Assistant: The quality of the image is " \
    text = "User: Rate the quality of the image <ImageHere> .  " \
        # with torch.cuda.amp.autocast():
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    img_embeds = model.encode_img(images)
    # print(img_embeds.shape)
    prompt_segs = text.split('<ImageHere>')
    prompt_seg_tokens = [
        model.tokenizer(seg,
                        return_tensors='pt',
                        add_special_tokens=i == 0).
        to(model.internlm_model.model.embed_tokens.weight.device).input_ids
        for i, seg in enumerate(prompt_segs)
    ]
    prompt_seg_embs = [
        model.internlm_model.model.embed_tokens(seg).expand(
            img_embeds.shape[0], -1, -1)
        for seg in prompt_seg_tokens
    ]
    prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
    prompt_embs = torch.cat(prompt_seg_embs, dim=1)
    outputs = model.internlm_model(
        inputs_embeds=prompt_embs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    # print(hidden_states.shape)
    # hidden state :  33 tensor tuple
    # hidden states [-1] shape 64*78*4096 -> input to trainable attention class module
    # apply self attention to aggregate to get 64*4096 vector
    return hidden_states
    # return hidden_states, outputs


def init_prompt(model):
    text = "User: Rate the quality of the image "
    prompt_tokens = model.tokenizer(text,
                                    return_tensors='pt',
                                    add_special_tokens=True).to(model.internlm_model.model.embed_tokens.weight.device).input_ids
    prompt_embs = model.internlm_model.model.embed_tokens(prompt_tokens).to(
        model.internlm_model.model.embed_tokens.weight.device)
    prompt_embs = prompt_embs.to(torch.float32)
    return prompt_embs


def get_init_logits_coop(model, images, learnable_prompt_embeddings):
    # report the int() usage because period is before image
    # text = "User: Rate the quality of the image <ImageHere> . Assistant: The quality of the image is " \
    text = "<ImageHere> .  " \
        # with torch.cuda.amp.autocast():
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    img_embeds = model.encode_img(images)
    # print(img_embeds.shape)
    prompt_segs = text.split('<ImageHere>')
    prompt_seg_tokens = [
        model.tokenizer(seg,
                        return_tensors='pt',
                        add_special_tokens=i == 0).
        to(model.internlm_model.model.embed_tokens.weight.device).input_ids
        for i, seg in enumerate(prompt_segs)
    ]
    prompt_seg_embs = [
        model.internlm_model.model.embed_tokens(seg).expand(
            img_embeds.shape[0], -1, -1)
        for seg in prompt_seg_tokens
    ]
    prompt_seg_embs = [learnable_prompt_embeddings.expand(img_embeds.shape[0], -1, -1),
                       img_embeds, prompt_seg_embs[-1]]
    prompt_embs = torch.cat(prompt_seg_embs, dim=1)
    outputs = model.internlm_model(
        inputs_embeds=prompt_embs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    # print(hidden_states.shape)
    # hidden state :  33 tensor tuple
    # hidden states [-1] shape 64*78*4096 -> input to trainable attention class module
    # apply self attention to aggregate to get 64*4096 vector
    return hidden_states
    # return hidden_states, outputs


def get_init_logits_no_llm(model, images):
    # report the int() usage because period is before image
    # text = "User: Rate the quality of the image <ImageHere> . Assistant: The quality of the image is " \
    text = "User: Rate the quality of the image <ImageHere> .  " \
        # with torch.cuda.amp.autocast():
    # print("model.query_tokens before")
    # print(model.query_tokens)
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    img_embeds = model.encode_img(images)
    # print(model.query_tokens)
    # print("model.query_tokens after")
    # print(model.query_tokens)
    return img_embeds


def get_init_logits_internlm_v2(model, images):
    # report the int() usage because period is before image
    text = "User: Rate the quality of the image <ImageHere> .  " \
        # with torch.cuda.amp.autocast():
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    img_embeds = model.encode_img(images)
    # print(img_embeds.shape)
    prompt_segs = text.split('<ImageHere>')
    prompt_seg_tokens = [
        model.tokenizer(seg,
                        return_tensors='pt',
                        add_special_tokens=i == 0).input_ids.to(model.get_input_embeddings().weight.device)
        for i, seg in enumerate(prompt_segs)
    ]
    prompt_seg_embs = [
        model.get_input_embeddings()(seg).expand(
            img_embeds.shape[0], -1, -1)
        for seg in prompt_seg_tokens
    ]
    prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
    prompt_embs = torch.cat(prompt_seg_embs, dim=1)
    outputs = model(input_ids=None, attention_mask=None, inputs_embeds=prompt_embs,
                    labels=None, output_attention=False, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states[-1]
    # print(hidden_states.shape)
    # hidden state :  33 tensor tuple
    # hidden states [-1] shape 64*78*4096 -> input to trainable attention class module
    # apply self attention to aggregate to get 64*4096 vector
    return hidden_states
