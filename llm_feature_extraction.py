# Code for: Getting output logits necessary for backprop from the model

import torch
import os

# llava imports


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
    text = "User: Rate the quality of the image <ImageHere> .  " \
        # with torch.cuda.amp.autocast():
    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    # print(f"shape of images before embedding: {images.shape}")
    img_embeds = model.encode_img(images)
    # print(f"shape of images after embedding : {images.shape}")

    prompt_segs = text.split('<ImageHere>')
    # print(f" the segmented prompts are : {prompt_segs}")
    prompt_seg_tokens = [
        model.tokenizer(seg,
                        return_tensors='pt',
                        add_special_tokens=i == 0).
        to(model.internlm_model.model.embed_tokens.weight.device).input_ids
        for i, seg in enumerate(prompt_segs)
    ]
    # print(f" the shape of tokenised segmented prompts is : {len(prompt_seg_tokens)}")
    # print(f" the first tokensed segment is : {prompt_seg_tokens[0]}")
    # print(f" the second tokensed segment is : {prompt_seg_tokens[1]}")

    prompt_seg_embs = [
        model.internlm_model.model.embed_tokens(seg).expand(
            img_embeds.shape[0], -1, -1)
        for seg in prompt_seg_tokens
    ]
    # print(f" length of tokenised segmented prompts so that it's dimensions are compatible for batch processing : {len(prompt_seg_embs)}")
    # print(f" shape of 2nd tokenised segmented prompts so that it's dimensions are compatible for batch processing : {prompt_seg_embs[0].shape}")
    # print(f" shape of 3rd tokenised segmented prompts so that it's dimensions are compatible for batch processing : {prompt_seg_embs[1].shape}")

    prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
    # is prompt_seg_embs converted from tensor to list here ?
    # place print statements which verify the above
    # place print statements every for datatype in this code
    # place print statements which give more relevant information
    # print(f" the length of prompt_seg_embs : {len(prompt_seg_embs)} ")
    # print(f" the shape of first element of prompt_seg_embs : {prompt_seg_embs[0].shape}")
    # print(f" the shape of second element of prompt_seg_embs : {prompt_seg_embs[1].shape}")
    # print(f" the shape of third element of prompt_seg_embs : {prompt_seg_embs[2].shape}")

    prompt_embs = torch.cat(prompt_seg_embs, dim=1)
    # print(f" the shape of after concatenating elements of prompt_seg_embs : {prompt_embs.shape}")

    outputs = model.internlm_model(
        inputs_embeds=prompt_embs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    # print(f"the shape of last hidden state is : {hidden_states.shape}")
    return hidden_states


def get_init_logits_internlm_v2(model, images):
    text = "User: Rate the quality of the image <ImageHere> .  " \
        # with torch.cuda.amp.autocast():
    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    print("printing image shapes")
    print(images.shape)

    img_embeds = model.encode_img(images)

    print("shape after encoding")
    print(img_embeds.shape)

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

    print("printing last hidden state shape")
    print(hidden_states.shape)

    return hidden_states


def get_init_logits_internlm_quantised(model, images, default_device):
    text = "User: Rate the quality of the image <ImageHere> .  " \
        # with torch.cuda.amp.autocast():
    images = images.to(default_device).half()
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    img_embeds = model.encode_img(images).to(default_device).half()
    prompt_segs = text.split('<ImageHere>')
    prompt_seg_tokens = [
        model.tokenizer(seg,
                        return_tensors='pt',
                        add_special_tokens=i == 0).input_ids.to(default_device)
        for i, seg in enumerate(prompt_segs)
    ]
    prompt_seg_embs = [
        model.get_input_embeddings()(seg).expand(
            img_embeds.shape[0], -1, -1).to(default_device).half()
        for seg in prompt_seg_tokens
    ]
    prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
    prompt_embs = torch.cat(prompt_seg_embs, dim=1).to(default_device).half()
    prompt_embs = prompt_embs.half().to(default_device)

    outputs = model(input_ids=None, attention_mask=None, inputs_embeds=prompt_embs,
                    labels=None, output_attention=False, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states[-1]
    return hidden_states


def get_init_logits_llava(model, images, tokenizer, image_processor):
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    text = "User: Rate the quality of the image "
    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    # with torch.cuda.amp.autocast():

    # print(f"shape of images before embedding: {images.shape}")
    # images = images.half()
    # print(images.dtype)
    images = model.encode_images(images)
    # print(f"shape of images after embedding: {images.shape}")

    text = text + DEFAULT_IMAGE_TOKEN + " . "
    # print(text) # User: Rate the quality of the image <image> .

    # print(f" the image token index is : {IMAGE_TOKEN_INDEX} ")
    input_ids = tokenizer_image_token(
        text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    # print(f" the shape of input_ids is : {input_ids.shape} ")
    # print(f" length of input_ids is : {len(input_ids)}")
    # print(f"  input_ids[0] shape : {input_ids[0].shape}")
    # print(f" values of input_ids : {input_ids}")

    # setting the device so all tensors are on the same GPU
    device = images.device
    input_ids = input_ids.to(device)

    image_token_index = (input_ids == IMAGE_TOKEN_INDEX).nonzero(
        as_tuple=True)[1].item()
    # print(image_token_index)
    text_segment_1 = input_ids[:, :image_token_index]
    text_segment_2 = input_ids[:, image_token_index + 1:]

    # embedding segments
    text_segment_1_emb = model.get_model().embed_tokens(
        text_segment_1).expand(images.shape[0], -1, -1).to(device)
    text_segment_2_emb = model.get_model().embed_tokens(
        text_segment_2).expand(images.shape[0], -1, -1).to(device)

    # print(f" shape of text_segment_1_emb : { text_segment_1_emb.shape }")
    # print(f" shape of text_segment_2_emb : { text_segment_2_emb.shape }")
    # print(text_segment_1_emb.shape)
    # print(images.shape)
    combined_embedding = torch.cat(
        [text_segment_1_emb, images, text_segment_2_emb], dim=1)
    # print(f" shape of concatenating embeddings : {combined_embedding.shape}")
    outputs = model(inputs_embeds=combined_embedding,
                    output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]

    # print("printing last hidden state shape")
    # print(last_hidden_state.shape)
    return last_hidden_state


def get_init_logits_mplug_owl(model, images, tokenizer, image_processor):
    from mplug_owl2.mm_utils import tokenizer_image_token
    from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    text = "User: Rate the quality of the image "
    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    # print(f"shape of images before embedding: {images.shape}")

    # print(images.dtype)
    # print(f"shape of images after embedding: {images.shape}")
    text = text + DEFAULT_IMAGE_TOKEN + " . "
    # print(text)  # User: Rate the quality of the image <image> .

    # print(f" the image token index is : {IMAGE_TOKEN_INDEX} ")
    input_ids = tokenizer_image_token(
        text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    # print(f" the shape of input_ids is : {input_ids.shape} ")
    # print(f" length of input_ids is : {len(input_ids)}")
    # print(f"  input_ids[0] shape : {input_ids[0].shape}")
    # print(f" values of input_ids : {input_ids}")

    # setting the device so all tensors are on the same GPU
    device = images.device
    input_ids = input_ids.to(device)
    batch_size = images.shape[0]
    # print(f"batch size : {batch_size}")
    input_ids_expanded = input_ids.expand(batch_size, -1)
    # print(f" after expanding input_ids : {input_ids_expanded}")
    # print(f" input_ids.shape[1] : { input_ids_expanded.shape[1]}")
    outputs = model(input_ids=input_ids_expanded, images=images,
                    output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]

    # print("printing last hidden state shape")
    # print(last_hidden_state.shape)

    # print("printing last hidden state shape")
    # print(last_hidden_state.shape)
    return last_hidden_state

def get_init_logits_no_qformer(model, images):
    # report the int() usage because period is before image
    # text = "User: Rate the quality of the image <ImageHere> . Assistant: The quality of the image is " \
    text = "User: Rate the quality of the image <ImageHere> .  " \
        # with torch.cuda.amp.autocast():

    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    # for name, param in model.named_parameters():
    #     print(f"Layer: {name}, Device: {param.device}")

    img_embeds = model.ln_vision(model.visual_encoder(images))

    return img_embeds