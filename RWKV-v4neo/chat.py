########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

print('Loading...')
from src.model_run import RWKV_RNN
import numpy as np
import os, copy, types, gc, sys
import torch
from rwkv.utils import PIPELINE
from prompt_toolkit import prompt

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)

CHAT_LEN_SHORT = 40
CHAT_LEN_LONG = 150
FREE_GEN_LEN = 256
GEN_alpha_presence = 0.4
GEN_alpha_frequency = 0.4  # Frequency Penalty
GEN_penalty_decay = 0.996
AVOID_REPEAT = '，：？！、'

args = types.SimpleNamespace()
args.RUN_DEVICE = "cuda"  # 'cpu' (already very fast) // 'cuda'
args.FLOAT_MODE = "fp16"  # fp32 (good for CPU) // fp16 (recommended for GPU) // bf16 (less accurate)
args.vocab_size = 50277
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0

# Modify this to use LoRA models; lora_r = 0 will not use LoRA weights.
args.MODEL_LORA = "../../finetuned/rwkv-5"
args.lora_r = 8
args.lora_alpha = 16


CHAT_LANG == 'Japanese'
args.MODEL_NAME = "../../world-World-7bjpn"
args.n_layer = 32
args.n_embd = 4096
args.ctx_len = 4096

print(f'loading... {args.MODEL_NAME}')
model = RWKV_RNN(args)
if 'world/' in args.MODEL_NAME or '-World-' in args.MODEL_NAME:
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    END_OF_TEXT = 0
    END_OF_LINE = 11

model_tokens = []

current_state = None


########################################################################################################

def run_rnn(tokens, newline_adj=0):
    global model_tokens, current_state
    for i in range(len(tokens)):
        model_tokens += [int(tokens[i])]
        if i == len(tokens) - 1:
            out, current_state = model.forward(model_tokens, current_state)
        else:
            current_state = model.forward(model_tokens, current_state, preprocess_only=True)

    # print(f'### model ###\n[{tokenizer.tokenizer.decode(model_tokens)}]')

    out[0] = -999999999  # disable <|endoftext|>
    out[187] += newline_adj
    # if newline_adj > 0:
    #     out[15] += newline_adj / 2 # '.'
    return out


all_state = {}


def save_all_stat(srv, name, last_out):
    n = f'{name}_{srv}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(current_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)


def load_all_stat(srv, name):
    global model_tokens, current_state
    n = f'{name}_{srv}'
    current_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']


def fix_tokens(tokens):
    if 'world/' in args.MODEL_NAME or '-World-' in args.MODEL_NAME:
        return tokens
    if len(tokens) > 0 and tokens[-1] == END_OF_LINE_DOUBLE:
        tokens = tokens[:-1] + [END_OF_LINE, END_OF_LINE]
    return tokens


########################################################################################################

# Run inference
print(f'\nRun prompt...')

# user, bot, interface, init_prompt = load_prompt(PROMPT_FILE)

interface = ":"
user = "Bob"
bot = "Alice"

# If you modify this, make sure you have newlines between user and bot words too

init_prompt = f'''
以下は、{bot}という女の子とその友人{user}の間で行われた会話です。 \
{bot}はとても賢く、想像力があり、友好的です。 \
{bot}は{user}に反対することはなく、{bot}は{user}に質問するのは苦手です。 \
{bot}は{user}に自分のことや自分の意見をたくさん伝えるのが好きです。 \
{bot}はいつも{user}に親切で役に立つ、有益なアドバイスをしてくれます。

{user}{interface} こんにちは{bot}、調子はどうですか？

{bot}{interface} こんにちは！元気ですよ。あたなはどうですか？

{user}{interface} 元気ですよ。君に会えて嬉しいよ。見て、この店ではお茶とジュースが売っているよ。

{bot}{interface} 本当ですね。中に入りましょう。大好きなモカラテを飲んでみたいです！

{user}{interface} モカラテって何ですか？

{bot}{interface} モカラテはエスプレッソ、ミルク、チョコレート、泡立てたミルクから作られた飲み物です。香りはとても甘いです。

{user}{interface} それは美味しそうですね。今度飲んでみます。しばらく私とおしゃべりしてくれますか？

{bot}{interface} もちろん！ご質問やアドバイスがあれば、喜んでお答えします。専門的な知識には自信がありますよ。どうぞよろしくお願いいたします！
'''

out = run_rnn(fix_tokens(pipeline.encode(init_prompt)))
save_all_stat('', 'chat_init', out)

gc.collect()
torch.cuda.empty_cache()

save_all_stat('', 'chat_init', out)

srv_list = ['dummy_server']
for s in srv_list:
    save_all_stat(s, 'chat', out)

print(f'### prompt ###\n[{pipeline.decode(model_tokens)}]\n')


def reply_msg(msg):
    print(f'{bot}{interface} {msg}\n')


def on_message(message):
    #     global model_tokens, current_state
    global model_tokens, model_state, user, bot, interface, init_prompt

    srv = 'dummy_server'

    msg = message.replace('\\n', '\n').strip()
    #     if len(msg) > 1000:
    #         reply_msg('your message is too long (max 1000 tokens)')
    #         return

    x_temp = 1.0
    x_top_p = 0.85
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp=" + f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p=" + f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")
    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5
    if x_top_p <= 0:
        x_top_p = 0
    msg = msg.strip()

    if msg == '+reset':
        out = load_all_stat('', 'chat_init')
        save_all_stat(srv, 'chat', out)
        reply_msg("Chat reset.")
        return

    # use '+prompt {path}' to load a new prompt
    elif msg[:8].lower() == '+prompt ':
        print("Loading prompt...")
        try:
            PROMPT_FILE = msg[8:].strip()
            user, bot, interface, init_prompt = load_prompt(PROMPT_FILE)
            out = run_rnn(fix_tokens(pipeline.encode(init_prompt)))
            save_all_stat(srv, 'chat', out)
            print("Prompt set up.")
            gc.collect()
            torch.cuda.empty_cache()
        except:
            print("Path error.")

    elif msg[:5].lower() == '+gen ' or msg[:3].lower() == '+i ' or msg[
                                                                   :4].lower() == '+qa ' or msg.lower() == '+++' or msg.lower() == '++':
        if msg[:5].lower() == '+gen ':
            new = '\n' + msg[5:].strip()
            # print(f'### prompt ###\n[{new}]')
            current_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:3].lower() == '+i ':
            msg = msg[3:].strip().replace('\r\n', '\n').replace('\n\n', '\n')
            new = f'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.

# Instruction:
{msg}

# Response:
'''
            #             print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qq ':
            new = '\nQ: ' + msg[4:].strip() + '\nA:'
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qa ':
            out = load_all_stat('', 'chat_init')

            real_msg = msg[4:].strip()
            new = f"{user}{interface} {real_msg}\n\n{bot}{interface}"
            # print(f'### qa ###\n[{new}]')

            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg.lower() == '+++':
            try:
                out = load_all_stat(srv, 'gen_1')
                save_all_stat(srv, 'gen_0', out)
            except:
                return

        elif msg.lower() == '++':
            try:
                out = load_all_stat(srv, 'gen_0')
            except:
                return

        begin = len(model_tokens)
        out_last = begin
        occurrence = {}
        for i in range(FREE_GEN_LEN + 100):
            for n in occurrence:
                out[n] -= (GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency)
            token = pipeline.sample_logits(
                out,
                temperature=x_temp,
                top_p=x_top_p,
            )
            if token == END_OF_TEXT:
                break
            for xxx in occurrence:
                occurrence[xxx] *= GEN_penalty_decay
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            if msg[:4].lower() == '+qa ':  # or msg[:4].lower() == '+qq ':
                out = run_rnn([token], newline_adj=-2)
            else:
                out = run_rnn([token])

            xxx = pipeline.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx:  # avoid utf-8 display issues
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
                if i >= FREE_GEN_LEN:
                    break
        print('\n')
        # send_msg = tokenizer.tokenizer.decode(model_tokens[begin:]).strip()
        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'gen_1', out)

    else:
        if msg.lower() == '+':
            try:
                out = load_all_stat(srv, 'chat_pre')
            except:
                return
        else:
            out = load_all_stat(srv, 'chat')
            msg = msg.strip().replace('\r\n', '\n').replace('\n\n', '\n')
            new = f"{user}{interface} {msg}\n\n{bot}{interface}"
            # print(f'### add ###\n[{new}]')
            out = run_rnn(pipeline.encode(new), newline_adj=-999999999)
            save_all_stat(srv, 'chat_pre', out)

        begin = len(model_tokens)
        out_last = begin
        print(f'{bot}{interface}', end='', flush=True)
        occurrence = {}
        for i in range(999):
            if i <= 0:
                newline_adj = -999999999
            elif i <= CHAT_LEN_SHORT:
                newline_adj = (i - CHAT_LEN_SHORT) / 10
            elif i <= CHAT_LEN_LONG:
                newline_adj = 0
            else:
                newline_adj = min(3, (i - CHAT_LEN_LONG) * 0.25)  # MUST END THE GENERATION

            for n in occurrence:
                out[n] -= (GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency)
            token = pipeline.sample_logits(
                out,
                temperature=x_temp,
                top_p=x_top_p,
            )
            for xxx in occurrence:
                occurrence[xxx] *= GEN_penalty_decay
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            out = run_rnn([token], newline_adj=newline_adj)
            out[END_OF_TEXT] = -999999999  # disable <|endoftext|>

            xxx = pipeline.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx:
                print(xxx, end='', flush=True)
                out_last = begin + i + 1

            send_msg = pipeline.decode(model_tokens[begin:])
            if '\n\n' in send_msg:
                send_msg = send_msg.strip()
                break

            # send_msg = tokenizer.tokenizer.decode(model_tokens[begin:]).strip()
            # if send_msg.endswith(f'{user}{interface}'): # warning: needs to fix state too !!!
            #     send_msg = send_msg[:-len(f'{user}{interface}')].strip()
            #     break
            # if send_msg.endswith(f'{bot}{interface}'):
            #     send_msg = send_msg[:-len(f'{bot}{interface}')].strip()
            #     break

        # print(f'{model_tokens}')
        # print(f'[{tokenizer.tokenizer.decode(model_tokens)}]')

        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'chat', out)


########################################################################################################

if CHAT_LANG == 'English':
    HELP_MSG = '''Commands:
say something --> chat with bot. use \\n for new line.
+ --> alternate chat reply
+reset --> reset chat

+gen YOUR PROMPT --> free single-round generation with any prompt. use \\n for new line.
+i YOUR INSTRUCT --> free single-round generation with any instruct. use \\n for new line.
+++ --> continue last free generation (only for +gen / +i)
++ --> retry last free generation (only for +gen / +i)

Now talk with the bot and enjoy. Remember to +reset periodically to clean up the bot's memory. Use RWKV-4 14B (especially https://huggingface.co/BlinkDL/rwkv-4-raven) for best results.
'''
elif CHAT_LANG == 'Japanese':
    HELP_MSG = f'''コマンド:
直接入力 --> ボットとチャットする．改行には\\nを使用してください．
+ --> ボットに前回のチャットの内容を変更させる．
+reset --> 対話のリセット．メモリをリセットするために，+resetを定期的に実行してください．

+i インストラクトの入力 --> チャットの文脈を無視して独立した質問を行う．改行には\\nを使用してください．
+gen プロンプトの生成 --> チャットの文脈を無視して入力したプロンプトに続く文章を出力する．改行には\\nを使用してください．
+++ --> +gen / +i の出力の回答を続ける．
++ --> +gen / +i の出力の再生成を行う.

ボットとの会話を楽しんでください。また、定期的に+resetして、ボットのメモリをリセットすることを忘れないようにしてください。
'''

print(HELP_MSG)
print(f'{CHAT_LANG} - {args.MODEL_NAME} - {args.RUN_DEVICE} - {args.FLOAT_MODE}')

print(f'{pipeline.decode(model_tokens)}'.replace(f'\n\n{bot}', f'\n{bot}'), end='')

########################################################################################################

while True:
    msg = prompt(f'{user}{interface} ')
    if len(msg.strip()) > 0:
        on_message(msg)
    else:
        print('Erorr: please say something')
