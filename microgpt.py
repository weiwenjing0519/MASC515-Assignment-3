import os
import math
import random
random.seed(42)

# --- Data Loading ---
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1

# --- Autograd Value Class ---
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data, self.grad = data, 0
        self._children, self._local_grads = children, local_grads
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    # 1. GELU Implementation (Gaussian Error Linear Units)
    def gelu(self):
        x = self.data
        # Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        out_data = 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
        # Simplified local gradient for the assignment
        local_grad = 0.5 * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
        return Value(out_data, (self,), (local_grad,))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v); [build_topo(c) for c in v._children]; topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad

# --- Hyperparameters ---
n_layer, n_embd, block_size, n_head, lora_r = 1, 16, 16, 4, 2
head_dim = n_embd // n_head

# --- 2. Parameter Initialization with LoRA & MoE ---
matrix = lambda nout, nin: [[Value(random.gauss(0, 0.08)) for _ in range(nin)] for _ in range(nout)]

def lora_init(nout, nin, r):
    return {'W': matrix(nout, nin), 'A': matrix(r, nin), 'B': matrix(nout, r)}

state_dict = {'wte': matrix(vocab_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'l{i}.qkv'] = lora_init(3 * n_embd, n_embd, lora_r)
    state_dict[f'l{i}.wo'] = lora_init(n_embd, n_embd, lora_r)
    # 4. Mixture of Experts (MoE) - 4 experts
    state_dict[f'l{i}.moe_gate'] = matrix(4, n_embd)
    for e in range(4):
        state_dict[f'l{i}.exp{e}.fc1'] = matrix(2 * n_embd, n_embd)
        state_dict[f'l{i}.exp{e}.fc2'] = matrix(n_embd, 2 * n_embd)

params = []
def collect(d):
    for v in d.values():
        if isinstance(v, dict): collect(v)
        else: [params.extend(row) for row in v]
collect(state_dict)

# --- Logic Modules ---

# LoRA Linear: y = Wx + (B @ A)x
def linear_lora(x, lora):
    std = [sum(wi * xi for wi, xi in zip(row, x)) for row in lora['W']]
    lora_path = [sum(bi * axi for bi, axi in zip(row, [sum(ai * xi for ai, xi in zip(arow, x)) for arow in lora['A']])) for row in lora['B']]
    return [s + l for s, l in zip(std, lora_path)]

# 3. RoPE (Rotary Positional Embedding)
def apply_rope(x_h, pos):
    res = []
    for j in range(0, head_dim, 2):
        theta = pos * (10000 ** (-j / head_dim))
        cos, sin = math.cos(theta), math.sin(theta)
        res.extend([x_h[j] * cos - x_h[j+1] * sin, x_h[j] * sin + x_h[j+1] * cos])
    return res

def softmax(logits):
    exps = [(v - max(l.data for l in logits)).exp() for v in logits]
    sum_e = sum(exps)
    return [e / sum_e for e in exps]

def rmsnorm(x):
    ss = (sum(xi * xi for xi in x) / len(x) + 1e-5) ** -0.5
    return [xi * ss for xi in x]

# --- GPT Forward Pass ---
def gpt(token_id, pos_id, keys, values):
    x = state_dict['wte'][token_id]
    for li in range(n_layer):
        # Attention Layer
        nx = rmsnorm(x)
        qkv = linear_lora(nx, state_dict[f'l{li}.qkv'])
        q, k, v = qkv[:n_embd], qkv[n_embd:2*n_embd], qkv[2*n_embd:]
        
        # RoPE application and Attention
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = apply_rope(q[hs:hs+head_dim], pos_id)
            k_h = apply_rope(k[hs:hs+head_dim], pos_id)
            if len(keys[li]) <= h: keys[li].append([]); values[li].append([])
            keys[li][h].append(k_h); values[li][h].append(v[hs:hs+head_dim])
            
            sim = [sum(q_h[j] * kh[j] for j in range(head_dim)) / head_dim**0.5 for kh in keys[li][h]]
            w = softmax(sim)
            x_attn.extend([sum(w[t] * values[li][h][t][j] for t in range(len(w))) for j in range(head_dim)])
        
        x = [xi + ri for xi, ri in zip(linear_lora(x_attn, state_dict[f'l{li}.wo']), x)]
        
        # MoE Layer (GELU inside experts)
        nx = rmsnorm(x)
        g_w = softmax([sum(wi * nxi for wi, nxi in zip(row, nx)) for row in state_dict[f'l{li}.moe_gate']])
        moe_out = [Value(0) for _ in range(n_embd)]
        for e in range(4):
            e_out = [sum(wi * nxi for wi, nxi in zip(row, nx)) for row in state_dict[f'l{li}.exp{e}.fc1']]
            e_out = [eo.gelu() for eo in e_out] # Using GELU
            e_out = [sum(wi * eoi for wi, eoi in zip(row, e_out)) for row in state_dict[f'l{li}.exp{e}.fc2']]
            moe_out = [m + g_w[e] * eo for m, eo in zip(moe_out, e_out)]
        x = [xi + mi for xi, mi in zip(x, moe_out)]

    return linear_lora(x, {'W': state_dict['lm_head'], 'A': [[0]*n_embd]*lora_r, 'B': [[0]*vocab_size]*lora_r})

# --- Training Loop ---
for step in range(200):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        logits = gpt(tokens[pos_id], pos_id, keys, values)
        losses.append(-softmax(logits)[tokens[pos_id+1]].log())
    loss = sum(losses) * (1/n)
    loss.backward()
    for p in params:
        p.data -= 0.01 * p.grad # Simple SGD
        p.grad = 0
    print(f"Step {step+1} | Loss: {loss.data:.4f}", end='\r')
