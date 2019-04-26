# coding=utf-8
'''

何晓明 2019/2/7
预处理Masked双向编码器和预测用户下一次行为的数据

'''
import random
import collections
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", './Data/50000/train_set.txt',
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", './Data/50000/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_file", './tmp/tf_examples.tfrecord',
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_integer('dupe_factor', 5, "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("short_seq_prob", 0.1, "Probability of creating sequences which are shorter than the maximum length.")

 
##################用来将vocab对应成数字####################
#这个代码可以和打开文件的代码合并起来
class FullTokenizer():
    
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_dt = self.reflect_vocab()
        
    def reflect_vocab(self):
        self.vocab_dt = collections.OrderedDict()
        index = 0
        for i in range(len(self.vocab)):
            self.vocab_dt[self.vocab[i]] = index
            index +=1
        return self.vocab_dt
    
    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.vocab_dt[token])
        return ids
##################创建训练实例##################
class TrainingInstance(object):

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, next_item, next_label):
    self.tokens = tokens
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels
    self.segment_ids = segment_ids
    self.next_item = next_item
    self.next_label = next_label

def create_training_instances(data, tokenizer,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    #?????????????????源代码中中是对input_files进行置乱?????????????????#
    rng.shuffle(data)
    vocab_words = list(tokenizer.vocab_dt.keys())
    instances = []
    #for _ in range(dupe_factor):
    for i in range(len(data)):
        
        instance = next_item_label(tokenizer, data[i],
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        if len(instance) > 1:
            for j in range(len(instance)):
                instances.append(instance[j])
        else:
            instances.append(instance[0])
    rng.shuffle(instances)
    return instances

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        cand_indexes.append(i)
    rng.shuffle(cand_indexes)
    
    output_tokens = list(tokens)

    masked_lm = collections.namedtuple('masked_lm', ['index', 'label'])

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens)*masked_lm_prob))))

    masked_lms = []
    convered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in convered_indexes:
            continue
        convered_indexes.add(index)
        
        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = '[Mask]'
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
        output_tokens[index] = masked_token
        
        masked_lms.append(masked_lm(index, label = tokens[index]))
        
    masked_lms = sorted(masked_lms, key = lambda x:x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels)
                
def next_item_label(tokenizer, data, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    #产生正样本和负样本
#    data_pos = data
#    data_neg = data
#    while True:
#        random_index = rng.randint(0, len(tokenizer.vocab_dt.keys()))
#        if vocab_words[random_index] not in data_pos:
#            break
#    data_neg[-1] = vocab_words[random_index]
    #添加segments_id字段
    segment_ids = [0 for i in range(len(data[0]))]
#    segment_ids.append(1)
    #打包正负样本
    data_pack = collections.namedtuple('data_pack', ['value', 'next_item', 'label'])
#    data_packs = []
#    data_packs.append(data_pack(value=data[0], next_item=data[1], label=data[2]))
    data_pack_ins = data_pack(value=data[0], next_item=data[1], label=data[2])
    #print('value is %s, next_item is %s, label is %s'%(data_pack_ins.value, data_pack_ins.next_item, data_pack_ins.label))
    #masked_lm变换并且写入
    instances = []
#    for i in range(len(data_packs)):
    (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(data_pack_ins.value, 
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
    instance = TrainingInstance(
            tokens = tokens,
            segment_ids = segment_ids,
            masked_lm_positions = masked_lm_positions,
            masked_lm_labels = masked_lm_labels,
            next_item = data_pack_ins.next_item,
            next_label = data_pack_ins.label)
   # print('instance中的值是：value:%s, next_item:%s, label:%s'%(instance.tokens, instance.next_item, instance.next_label))
    instances.append(instance)
    return instances

def write_instance_to_example_file(instances, tokenizer, max_seq_length, max_predictions_per_seq, output_files):
    print('2')
    writers = tf.python_io.TFRecordWriter(output_files)
    print('3')
    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        print(input_ids)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        hist_len = len(input_ids)
        next_item = tokenizer.convert_tokens_to_ids([instance.next_item])
        print('next_item is ',next_item)
        assert len(input_ids) <= max_seq_length
 
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)
        
        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)
            
        next_item_label = 1 if instance.next_label else 0
            
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        features["next_item_labels"] = create_int_feature([next_item_label])
        features['hist_len'] = create_int_feature([hist_len])
        features["next_item"] = create_int_feature(next_item)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        print('5')
        writers.write(tf_example.SerializeToString())
        print('6')
        total_written += 1
        print('7')
    writers.close()
    print('8')
    tf.logging.info("Wrote %d total instances", total_written)
    print('9')    
def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature

def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature

if __name__ == "__main__":
    
    tf.logging.set_verbosity(tf.logging.INFO)
    ##################读入数据##################
    with tf.gfile.GFile(FLAGS.input_file, 'r') as f:
        data = eval(f.read())
    ##如果数据长度超过max_seq_length,只取最大长度的数据
    length = []
    for i in range(len(data)):
        length.append(len(data[i][0]))
    print('最大长度是：', max(length))            
    ##################设置随机种子##################
    rng = random.Random(FLAGS.random_seed)
    
    with open(FLAGS.vocab_file, 'r', encoding='UTF-8') as f:
        vocab_ls = f.read()
        vocab_ls = vocab_ls.strip('\n').split('\n')
        
    tf.logging.set_verbosity(tf.logging.INFO)
    tokenizer = FullTokenizer(vocab_ls)

    instances = create_training_instances(data, tokenizer, FLAGS.dupe_factor,
                                          FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
                                          rng)
    
    print('实例的总数是：', len(instances))

    output_files = FLAGS.output_file
    tf.logging.info("*** Writing to output files ***")
    print('1')
    write_instance_to_example_file(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)




