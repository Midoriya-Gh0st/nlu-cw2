from string import punctuation, digits

"""
1. "total count" 和 "number" of word types 有什么区别?

2. 这个total-size是包括UNK的, 还是没有进行UNK处理之前的?
    [现在] = [原先] - [UNK] + 1
    
3. 什么是linguistic phenomenon (语言现象)?
    - 有['m]缩写;
    - 有相同stem的单词被处理为UNK, 比如: [write, wrote], [warm, warmer, warming], 但是这些都只是出现一次, 有什么影响?
        - [universities, university], [undergo, undergone]; - 大量的这种情况, 造成无法学习这种, "形态"上的变化, (语义, 时态等的改变)
        - 如何分词? 有个技术是, sub-word! 字节对编码! <- token
        - 但write/wrote, 无法解决;
    
    - welthungerhilfe, 外来词, 是德语; -> 和第四问更相关;
    - 有很多数字[日期/时间];  -- 这个不属于phenomena吧?
    
4. unique token, 是问被替换"UNK"之后, 还是最初的? 
    - 1,460 tokens, 相同, 但一半以上的, 被替换了;  [???]
    - 如何利用: 在BERT架构中, 对一些rare_word直接进行复制; => 将source中的rare_word, 来替换generate_unk;
            -- 比如number(年份, 时间..), 姓名, 组织名称, 这些在两种语言中, 不会发生改变; 
    - 如果是最初(即: 包含出现次数=1)的, 那利用: 不把这种[只出现一次, 但在两种lang都存]的单词进行"UNK"处理;
    - 对这种word, 能否提升 attention 权重?
    - 如何判断在source中是rare? ++ 

5. 鉴于上述观察，您认为 NMT 系统将如何受到两种语言的[sentence length]、[tokenization process]和[UNK]的影响？
    - 过长句子, 翻译质量不高; - 如果使用attention, 不会造成影响;
    - 
    - token的技术(方式)很重要, 粒度越小, 质量越好, 粒度越大, 比如直接分"整词", 会导致无法更好学习 "时态变化", "数量变化";
        -- 尽管lstm有这种学习词性的能力, 但不能灵活 "生成" (即: 泛化性能, 比如处理OOV); (trade-off, 太过细, 不好)
        -- beam search
        - token太坏, 很多词本来就一样的, 也给弄得不一样的了.
    - UNK: 会造成大量信息的损失, 这些主要是number/noun, 常常代表重要信息. 比如时间, 主要人物; => 可以直接复制这个句子里的单词.
        -- 不是说从所有的source复制, 而是从当前这个source_sentence复制, 比如来自德语的人名, 
        -- 什么时候复制: 当产生<UNK>的时候;
    - 
"""

def statistic_q2(lang: str):
    lang_words = []             # 所有单词
    lang_distinct_words = []    # unique word
    lang_dict = {}

    lang_f = open(f"../europarl_raw/train.{lang}", 'r', encoding='UTF-8')
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
    same_words = []
    for en_word in en_distinct_words:
        for de_word in de_distinct_words:
            if en_word == de_word:
                same_words.append(f'{en_word}\n')
                same_words_count += 1

    same_words = sorted(same_words)
    with open('same_words.txt', 'w', encoding='utf-8') as f:
        for word in same_words:
            f.write(word)

    print('same_words_count:', same_words_count)
