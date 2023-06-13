import torch
from transformers import BertTokenizer, BertForSequenceClassification

def predict(sentence):
    model = BertForSequenceClassification.from_pretrained('/root/sunyd/pretrained_models/bert-base-chinese/', num_labels=4)
    tokenizer = BertTokenizer.from_pretrained('/root/sunyd/pretrained_models/bert-base-chinese/')

    # 加载之前保存的状态字典
    model_state = torch.load('checkpoints/2023-06-13_06-44-09/TextClassify_epoch2.pth', map_location='cpu')
    model.load_state_dict(model_state)

    inputs = tokenizer(sentence, return_tensors='pt')

    y_pred = model(inputs['input_ids'], 
                    token_type_ids=None)

    logits = y_pred.logits
    preds = torch.argmax(logits, dim=1)
    return preds.detach().numpy()[0]
    
if __name__ == '__main__':
    labels = ['会议', '事件', '文献', '组织机构']
    test_sentence = "中央政治局会议于1953年5月18日在北京召开。毛泽东主持会议。会议讨论文化教育工作。刘少奇、周恩来在会上发言。毛泽东作了讲话。在谈到办学问题时说：办好学校，首先要解决学校的领导骨干问题，而且先要解决大学的领导骨干问题。在谈到教材编写问题时说：目前30个编辑太少了，增加到300人也不算多。宁可把别的摊子缩小点，必须抽调大批干部编写教材。确定补充150个编辑干部，由中央组织部负责解决配备。在谈到历史与语文教学问题时说：历史与语文应分开教学，责成组织委员会讨论解决。在谈到文字改革问题时说：第一个五年计划期间，要搞出简体字来，简体字可以创造。同时要研究注音字母，它有长期历史。将来拼音，要从汉字注音字母中搞出字母来。文字改革，第一步用简体字，注音字母，第二步拼音化。在谈到小学的整顿问题时说：“整顿巩固，保证质量，重点发展，稳步前进”的方针很好，但不要整过头了。"
    print('文档：{}'.format(test_sentence))
    print('这个文档的类别是：{}'.format(labels[predict(test_sentence)]))
    
