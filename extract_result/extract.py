import matplotlib.pyplot as plt


def read_content(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        train_loss, valid_loss, valid_perplexity = [], [], []
        lines = f.readlines()
        for line in lines:
            if line.strip() == "detect::end":
                print(line)
                break
            if len(line) >= 20 and line.split()[0] == 'Epoch':
                arr = line.split()
                if arr[2] == 'loss':  # train
                    train_loss.append(float(arr[3]))
                elif arr[2] == 'valid_loss':  # validation
                    valid_loss.append(float(arr[3]))
                    valid_perplexity.append(float(arr[-1]))
    return train_loss, valid_loss, valid_perplexity


q1_out_log = '../question/baseline-out.txt'
q1_tra_loss, q1_val_loss, q1_val_ppl = read_content(q1_out_log)

q2_out_log = '../question/q4-out.txt'
q2_tra_loss, q2_val_loss, q2_val_ppl = read_content(q2_out_log)

print(len(q1_tra_loss))  # 5.625 -> 2.142
print(len(q1_val_loss))  # 5.09 -> 3.3
print(len(q1_val_ppl))  # 163 -> 27

print(len(q2_tra_loss))  # 5.797 -> 2.323
print(len(q2_val_loss))  # 5.43 -> 3.39
print(len(q2_val_ppl))  # 226 -> 29.5


q1_title = "Q1: 1 encoder layer, 1 decoder layer"
q2_title = "Q4: 2 encoder layers, 3 decoder layers"


def my_plot(title, tra_loss, val_loss):
    plt.ylim([1.5, 6])
    plt.title(title)
    plt.plot(tra_loss)
    plt.plot(val_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(["train", "validation"])
    plt.show()
    


my_plot(q1_title, q1_tra_loss, q1_val_loss)
my_plot(q2_title, q2_tra_loss, q2_val_loss)

of_out_log = '../question/overfit-out2.txt'
of_tra_loss, of_val_loss, of_val_ppl = read_content(of_out_log)
my_plot("over-fitting", of_tra_loss, of_val_loss)

# q4:
# validation已经收敛, 但是train还在下降, 这说明train开始"复制数据", 即过拟合;
# 对于过拟合, 继续增加layer, 则会导致性能降低;
# 对于架构越复杂(越深), 模型性能越好? 不对,
# https://www.kdnuggets.com/2019/12/5-techniques-prevent-overfitting-neural-networks.html

# 奥卡姆剃刀: 相似性能, 选择最简单的.
# https://elitedatascience.com/overfitting-in-machine-learning#how-to-detect

# 但是必须那个loss变小又变大, 才算overfit, 这里只是趋于平缓, 也是吗?
# 检查后面十次输出, val_loss确实先减小, 又变大;

# https://datascience.stackexchange.com/questions/43191/validation-loss-is-not-decreasing
# https://zhuanlan.zhihu.com/p/136786657
# https://blog.ailemon.net/2019/02/26/solution-to-loss-doesnt-drop-in-nn-train/

print(q2_val_loss)

# - 这种变化对dev-ppl, test-BLEU, train-loss有什么影响(所有这些都与Q1中给出的baseline指标相比)?
#   -- dec-ppl增加 (变坏), test-BLEU下降 (变坏), train-loss[5.797->2.323], 相比[5.625->2.142], 也是增加 (变坏)
# - 你能解释为什么它在train, dev, test上比基线单层模型更差/更好吗？
#   -- 更差了. 本来就有过拟合, 应该增加数据量, drop-layer(以采用), 或者简化模型. 而增加了层数, 因此过拟合也更加严重了.
# - 训练集、开发集和测试集的性能有区别吗？ 为什么会这样？
#   -- ? 什么意思?

