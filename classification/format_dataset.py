import json
def format_choices(r:list[str]):
    out = []
    for i,c in enumerate(r):
        c = c.replace('A','').replace('B','').replace('C','').replace('D','').replace('.','').replace('．','').strip()
        out.append('ABCD'[i]+'. '+c)
    return '|'.join(out)
def cnt_sentence(s:str):
    def cnt_num(s:str,c:str):
        return len(s.split(c))-1
    return sum([cnt_num(s,c) for c in ['。','！','？']])
out_dirs = {
    '../Datasets/CLS/train.json':'../Datasets/CLS_formatted/train.json',
    '../Datasets/CLS/dev.json':'../Datasets/CLS_formatted/dev.json',
}
for c,out_dir in out_dirs.items():
    out_dict = []
    s = json.load(open(c))
    for item in s:
        content = item['Content']
        questions = item['Questions']
        for question in questions:
            question_text = question['Question']
            question_choices = question['Choices']
            question_ans = question['Answer']
            out_dict.append({
                "messages": [
                    {
                        "role": "user",
                        "content": f"""文段#{content}*句子个数#共{cnt_sentence(content)}句话*题目#{question_text}*选项#{format_choices(question_choices)}"""
},
                    {
                        "role": "assistant",
                        "content": question_ans
                    }
                ]
            })
    json.dump(out_dict,open(out_dir,'w',encoding='utf-8'),indent=4,ensure_ascii=False)