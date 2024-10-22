[TOC]
# ChatGLM-6B 单双向代码
模型地址:/ssd3/hehuizhang/llm/chatglmv1/ZhipuAI/ChatGLM-6B

## 1. 双向mask: modeling_chatglm.py
```
class ChatGLMPreTrainedModel(PreTrainedModel):
    ……
    def get_masks(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
        attention_mask.tril_()
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()

        return attention_mask

class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel):
    ……
    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1,
            do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):

```

## 2. 单向mask: modeling_chatglm_unidirection.py
```
class ChatGLMPreTrainedModel(PreTrainedModel):
    ……
    def get_masks(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
        attention_mask.tril_()
        # for i, context_length in enumerate(context_lengths):
        #     attention_mask[i, :, :context_length] = 1
        attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()

        return attention_mask


class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel):
    ……
    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = [("你好,你是谁", "我是一个人工智能助手，专门设计用来回答各种问题。")], max_length: int = 2048, num_beams=1,
             do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
```

# 评测
## 1、评测总览
1. 信息记录

| model | 数据集 | 起始时间 | 结束时间 | 耗时 | 设备 | 卡数 | cuda版本 | 服务器 |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| ChatGLM-6B bidirectional_mask(双向mask)| mmlu | 2024/10/16 12:56:44 | 2024/10/16 14:40:35 | 104(min）| NVIDIA GeForce RTX 3090 | 8 | 12.1 | 10.1.100.66 |
| ChatGLM-6B single_mask(单向mask) | mmlu | 2024/10/16 14:48:04 | 2024/10/16 10/16 18:51:44 | 243(min） | NVIDIA GeForce RTX 3090 | 8 | 12.1 | 10.1.100.66 |
| ChatGLM-6B bidirectional_mask(双向mask)| gsm8k | 2024/10/19 22:56:18 | 2024/10/20 01:12:03 | 136(min）| NVIDIA GeForce RTX 3090 | 8 | 12.1 | 10.1.100.66 |
| ChatGLM-6B single_mask(单向mask) | gsm8k | 2024/10/19 16:03:45 | 2024/10/20 10/16 19:09:04 | 186(min） | NVIDIA GeForce RTX 3090 | 8 | 12.1 | 10.1.100.66 |



2. **summary**

| model | 数据集 | metric | mode | 结果 |
| -----------| ----------- | ----------- | ----------- | ----------- |
| ChatGLM-6B bidirectional_mask(双向mask) | mmlu | naive_average | gen | 37.27 |
| ChatGLM-6B single_mask(单向mask) | mmlu | naive_average | gen | 23.22 |
| ChatGLM-6B bidirectional_mask(双向mask) | gsm8k | accuracy | gen | 5.23 |
| ChatGLM-6B single_mask(单向mask) | gsm8k | accuracy | gen | 0.68 |




## 2、评测详情
总结：单向mask测试效果比双向mask差很多.
    1. gsm8k数据集，双向mask评测结果5.23，单向mask评测结果0.68。
    2. mmlu数据集，双向mask评测结果37.27，单向mask评测结果23.22。mmlu数据集，57个评测类别，55个双向比单向accuracy大，2个单向accuracy比双向大,每个评测类别里面，很多question的prediction值，单向mask是比较长的句子，双向mask是单选a、b、c、d、
单向比双向accuracy最大的是lukaemon_mmlu_econometrics，值是-1.754
双向比单向accuracy最大的是lukaemon_mmlu_us_foreign_policy， 值是：38.0


| 数据集:mmlu | ChatGLM_6B_bidirectional_mask | ChatGLM_6B_single_mask | |
| -----------| ----------- | ----------- | ----------- |
| 不同类别的评测结果 | metric(accuracy) | metric(accuracy) | diff(双向-单向) |
| lukaemon_mmlu_econometrics | 24.561403508771928 | 26.31578947368421 |-1.7543859649122808 |
| lukaemon_mmlu_college_biology | 34.72222222222222 | 19.444444444444446 |15.277777777777775 |
| lukaemon_mmlu_professional_psychology | 35.78431372549019 | 20.588235294117645 |15.196078431372548 |
| lukaemon_mmlu_logical_fallacies | 46.012269938650306 | 22.085889570552148 |23.926380368098158 |
| lukaemon_mmlu_clinical_knowledge | 39.62264150943396 | 26.79245283018868 |12.830188679245282 |
| lukaemon_mmlu_college_computer_science | 38.0 | 16.0 |22.0 |
| lukaemon_mmlu_nutrition | 38.88888888888889 | 25.49019607843137 |13.398692810457522 |
| lukaemon_mmlu_college_medicine | 42.19653179190752 | 21.965317919075144 |20.231213872832374 |
| lukaemon_mmlu_human_aging | 42.600896860986545 | 25.56053811659193 |17.040358744394617 |
| lukaemon_mmlu_human_sexuality | 44.274809160305345 | 22.137404580152673 |22.137404580152673 |
| lukaemon_mmlu_high_school_government_and_politics | 49.22279792746114 | 25.906735751295333 |23.316062176165804 |
| lukaemon_mmlu_prehistory | 36.7283950617284 | 27.469135802469136 |9.259259259259263 |
| lukaemon_mmlu_miscellaneous | 45.21072796934866 | 24.393358876117496 |20.817369093231164 |
| lukaemon_mmlu_machine_learning | 33.035714285714285 | 27.67857142857143 |5.357142857142854 |
| lukaemon_mmlu_high_school_us_history | 38.72549019607843 | 25.98039215686275 |12.745098039215684 |
| lukaemon_mmlu_astronomy | 38.81578947368421 | 19.736842105263158 |19.078947368421055 |
| lukaemon_mmlu_sociology | 50.24875621890548 | 22.885572139303484 |27.363184079601993 |
| lukaemon_mmlu_high_school_microeconomics | 34.45378151260504 | 21.84873949579832 |12.60504201680672 |
| lukaemon_mmlu_public_relations | 40.0 | 18.181818181818183 |21.818181818181817 |
| lukaemon_mmlu_high_school_biology | 40.96774193548387 | 26.451612903225808 |14.516129032258064 |
| lukaemon_mmlu_abstract_algebra | 21.0 | 8.0 |13.0 |
| lukaemon_mmlu_high_school_physics | 27.81456953642384 | 20.52980132450331 |7.284768211920529 |
| lukaemon_mmlu_management | 42.71844660194174 | 37.86407766990291 |4.854368932038831 |
| lukaemon_mmlu_college_chemistry | 30.0 | 31.0 |-1.0 |
| lukaemon_mmlu_world_religions | 40.35087719298245 | 22.22222222222222 |18.12865497076023 |
| lukaemon_mmlu_high_school_statistics | 25.925925925925924 | 22.685185185185187 |3.240740740740737 |
| lukaemon_mmlu_international_law | 47.93388429752066 | 20.66115702479339 |27.272727272727273 |
| lukaemon_mmlu_professional_accounting | 32.269503546099294 | 22.69503546099291 |9.574468085106385 |
| lukaemon_mmlu_high_school_chemistry | 34.48275862068966 | 24.137931034482758 |10.3448275862069 |
| lukaemon_mmlu_high_school_european_history | 46.666666666666664 | 23.03030303030303 |23.636363636363633 |
| lukaemon_mmlu_global_facts | 27.0 | 14.000000000000002 |12.999999999999998 |
| lukaemon_mmlu_moral_scenarios | 24.35754189944134 | 20.446927374301676 |3.910614525139664 |
| lukaemon_mmlu_business_ethics | 46.0 | 26.0 |20.0 |
| lukaemon_mmlu_high_school_psychology | 50.27522935779817 | 23.669724770642205 |26.605504587155963 |
| lukaemon_mmlu_virology | 36.144578313253014 | 34.93975903614458 |1.2048192771084345 |
| lukaemon_mmlu_medical_genetics | 38.0 | 17.0 |21.0 |
| lukaemon_mmlu_moral_disputes | 38.4393063583815 | 25.14450867052023 |13.29479768786127 |
| lukaemon_mmlu_high_school_world_history | 46.835443037974684 | 22.78481012658228 |24.050632911392405 |
| lukaemon_mmlu_college_physics | 23.52941176470588 | 21.568627450980394 |1.9607843137254868 |
| lukaemon_mmlu_electrical_engineering | 37.93103448275862 | 24.82758620689655 |13.103448275862068 |
| lukaemon_mmlu_computer_security | 46.0 | 25.0 |21.0 |
| lukaemon_mmlu_professional_medicine | 26.838235294117645 | 25.36764705882353 |1.470588235294116 |
| lukaemon_mmlu_high_school_mathematics | 22.962962962962962 | 10.0 |12.962962962962962 |
| lukaemon_mmlu_philosophy | 38.90675241157556 | 27.652733118971064 |11.254019292604497 |
| lukaemon_mmlu_jurisprudence | 45.370370370370374 | 23.14814814814815 |22.222222222222225 |
| lukaemon_mmlu_elementary_mathematics | 25.132275132275133 | 23.28042328042328 |1.8518518518518547 |
| lukaemon_mmlu_high_school_computer_science | 28.000000000000004 | 22.0 |6.0000000000000036 |
| lukaemon_mmlu_high_school_macroeconomics | 38.71794871794872 | 27.17948717948718 |11.538461538461544 |
| lukaemon_mmlu_security_studies | 32.6530612244898 | 18.367346938775512 |14.285714285714285 |
| lukaemon_mmlu_high_school_geography | 45.45454545454545 | 23.737373737373737 |21.717171717171716 |
| lukaemon_mmlu_us_foreign_policy | 61.0 | 23.0 |38.0 |
| lukaemon_mmlu_conceptual_physics | 30.21276595744681 | 26.80851063829787 |3.4042553191489375 |
| lukaemon_mmlu_marketing | 59.82905982905983 | 29.48717948717949 |30.34188034188034 |
| lukaemon_mmlu_anatomy | 27.40740740740741 | 24.444444444444443 |2.9629629629629655 |
| lukaemon_mmlu_college_mathematics | 27.0 | 21.0 |6.0 |
| lukaemon_mmlu_formal_logic | 26.984126984126984 | 19.047619047619047 |7.936507936507937 |
| lukaemon_mmlu_professional_law | 30.182529335071706 | 23.92438070404172 |6.258148631029986 |