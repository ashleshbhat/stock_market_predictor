
def get_word_list_pos():
    pos_word_file = "pos.txt"
    with open(pos_word_file,"r") as file:
        pos_lines = file.read().split('\n')
        return pos_lines

def get_word_list_neg():
    pos_word_file = "neg.txt"
    with open(pos_word_file,"r") as file:
        neg_lines = file.read().split('\n')
        return neg_lines

def get_news_list():
    news_file = "sources.txt"
    with open(news_file,"r") as file:
        sources = file.read().split('\n')
        return sources

