from string import punctuation, digits

"""
1. "total count" 和 "number" of word types 有什么区别?

2. 这个total-size是包括UNK的, 还是没有进行UNK处理之前的?
    [现在] = [原先] - [UNK] + 1
    
3. 什么是linguistic phenomenon (语言现象)?
    - 有['m]缩写;
    - 有相同stem的单词被处理为UNK, 比如: [write, wrote], [warm, warmer, warming], 但是这些都只是出现一次, 有什么影响?
    - 有很多数字[日期/时间];
    
4. unique token, 是问被替换"UNK"之后, 还是最初的?
    - 如果是最初(即: 包含出现次数=1)的, 那利用: 不把这种[只出现一次, 但在两种lang都存]的单词进行"UNK"处理;
    - 对这种word, 能否提升权重?
"""

def statistic_q2(lang: str):
    lang_words = []             # 所有单词
    lang_distinct_words = []    # unique word
    lang_dict = {}

    lang_f = open(f"./europarl_raw/train.{lang}", 'r', encoding='UTF-8')
    content = lang_f.readlines()

    for line in content:
        line = line.strip('\r\n\t')
        line = line.split()

        for word in line:
            # if word == '' or word in punctuation:  # or word.isdigit():
            #     continue
            lang_words.append(word)
            if word not in lang_distinct_words:
                lang_distinct_words.append(word)
            if word not in lang_dict.keys():
                lang_dict[word] = 1
            else:
                lang_dict[word] += 1

    lang_unk = []
    for word, count in lang_dict.items():
        if count == 1:
            lang_unk.append(word)

    print(f'{lang}_words:', len(lang_words))
    print(f'{lang}_distinct_words:', len(lang_distinct_words))
    print(f'{lang}_unk:', len(lang_unk))
    print(f'{lang}_voc_size [+1]:', len(lang_distinct_words) - len(lang_unk) + 1)

    lang_unk.sort()
    with open(f'{lang}_unk.txt', 'w', encoding="utf-8") as lang_unk_f:
        for word in lang_unk:
            lang_unk_f.write(word + '\n')
    print()

    return lang_distinct_words


if __name__ == '__main__':
    en_distinct_words = statistic_q2('en')
    de_distinct_words = statistic_q2('de')

    same_words_count = 0
    for en_word in en_distinct_words:
        for de_word in de_distinct_words:
            if en_word == de_word:
                same_words_count += 1
    print('same_words_count:', same_words_count)
