def plot(texts:list[str],place:str):
    checkpoint = 0
    import re
    train_losses = {}
    valid_losses = {}
    pples = {}
    epoch = 0
    for cell in texts:
        print(cell)
        if 'Loss' in cell:
            # print(cell)
            epoch = int(re.findall(r'Epoch:\ (\d+)\,',cell)[0])
            train_loss = float(re.findall(r'Loss:\ (\d+\.\d+)\,',cell)[0])
            train_losses[epoch]=train_loss
        elif 'loss' in cell:
            ppl = float(re.findall(r'ppl:\ (\d+\.\d+)',cell)[0])
            valid_loss = float(re.findall(r'loss:\ (\d+\.\d+)\,',cell)[0])
            valid_losses[epoch]=valid_loss
            pples[epoch]=ppl
    if len(train_losses) != len(valid_losses):
        del train_losses[epoch]
    assert len(pples) == len(train_losses),(len(train_losses),len(pples))
    assert len(train_losses) == len(valid_losses),(len(valid_losses),len(train_losses))

    import numpy as np
    import matplotlib.pyplot as plt
    x = np.linspace(1,len(pples),len(pples))
    train_losses = np.array([train_losses[i] for i in x])
    valid_losses = np.array([valid_losses[i] for i in x])
    pples = np.array([pples[i] for i in x])
    N = 192
    train_losses = train_losses[:N]
    valid_losses = valid_losses[:N]
    pples = pples[:N]
    x = x[:N]
    open('./1.txt','w').write(str(train_losses.tolist()))
    plt.plot(x,train_losses/2500,label='train loss(/2500)')
    plt.plot(x,valid_losses,label='valid loss')
    plt.plot(x,pples/100,label='perplexity(/100)')
    plt.legend()
    # plt.show()
    plt.savefig(place)
if __name__ == '__main__':
    import json
    with open('example.ipynb', 'r') as file:
        notebook = json.load(file)

    outputs = []
    for cell in notebook['cells']:
        if 'outputs' in cell:
            outputs.extend(cell['outputs'])
    texts = []
    for output in outputs:
        if 'text' in output:
            texts.extend(output['text'])
    plot(texts,'./lstm_lm.png')