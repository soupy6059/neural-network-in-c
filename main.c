#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#define NOTE(X)
#define ROWS 0
#define COLS 1
#define TODO(X) do { assert(0 && (X)); } while(0)

typedef struct Matrix_ {
    void *creator;
    size_t shape[2];
    double *val;
    double *grad;
} Matrix;

void Matrix_ctor(Matrix *m, size_t *shape) {
    m->creator = NULL;
    m->shape[0] = shape[0];
    m->shape[1] = shape[1];
    m->val = calloc(shape[0] * shape[1], sizeof(*m->val));
    m->grad = calloc(shape[0] * shape[1], sizeof(*m->grad));
}

void Matrix_dtor(Matrix m) {
    free(m.val);
    free(m.grad);
}

double *Matrix_at(Matrix*,size_t,size_t);
void Matrix_cout(Matrix A) {
    for(size_t row = 0; row < A.shape[ROWS]; ++row) {
        printf("[");
        for(size_t col = 0; col < A.shape[COLS]; ++col) {
            printf(" %f", *Matrix_at(&A, row, col));
        }
        printf(" ]\n");
    }
}

void Matrix_set_copy(Matrix *this, double *val) {
    for(size_t i = 0; i < this->shape[0] * this->shape[1]; ++i) {
        this->val[i] = val[i];
    }
}
void Matrix_set_move(Matrix *this, double *val) {
    free(this->val);
    this->val = val;
}

size_t randi(size_t low, size_t high) {
    assert(high >= low);
    size_t dist = high - low;
    size_t R = rand() % dist;
    return R + low;
}

double randf() {
    int steps = 100000;
    int num = rand() % steps;
    return ((double)num) / ((double)steps);
}

double randf_normal() {
    size_t N = 30;
    double acc = 0.0;
    for(size_t k = 0; k < N; ++k) {
        acc += randf();
    }
    return acc / (double)N;
}

void Matrix_rand(Matrix *this) {
    for(size_t i = 0; i < this->shape[0] * this->shape[1]; ++i) {
        this->val[i] = randf();
    }
}


struct MatrixOperation_;
void MatrixOperation_zero_grad(struct MatrixOperation_ *this);
void Matrix_zero_grad(Matrix *this) {
    for(size_t i = 0; i < this->shape[0] * this->shape[1]; ++i) {
        this->val[i] = 0;
    }
    if(this->creator) MatrixOperation_zero_grad((struct MatrixOperation_*)this->creator);
}

void Matrix_backward(Matrix *this, void *s_) {
    Matrix *s = s_;
    assert(this->shape[0] == s->shape[0]
    && this->shape[1] == s->shape[1] && "bad s shape");
    for(size_t i = 0; i < this->shape[0] * this->shape[1]; ++i) {
        this->val[i] += s->val[i];
    }
}

size_t Matrix_length(Matrix *this) {
    return this->shape[0];
}

double *Matrix_call(Matrix *this) {
    return this->val;
}

void Matrix_apply(Matrix *dest, Matrix *src, double(*fn)(double)) {
    assert(dest->shape[0] == src->shape[0]
    && dest->shape[1] == src->shape[1] && "bad src|dest shape");
    for(size_t i = 0; i < dest->shape[0] * dest->shape[1]; ++i) {
        dest->val[i] = (*fn)(src->val[i]);
    }
}

Matrix *Matrix_mul(double *s, Matrix *A) {
    Matrix *src = malloc(sizeof(Matrix));
    Matrix_ctor(src, A->shape);
    Matrix_set_copy(src, A->val);
    for(size_t i = 0; i < src->shape[0] * src->shape[1]; ++i) {
        src->val[i] *= *s;
    }
    return src;
}

double *Matrix_at(Matrix *A, size_t i, size_t j) {
    return &A->val[i + A->shape[1] * j];
}

Matrix *Matrix_product(Matrix *A, Matrix *B) {
    Matrix *res = malloc(sizeof(Matrix));
    size_t res_shape[2] = {A->shape[ROWS], B->shape[COLS]};
    Matrix_ctor(res, res_shape);
    
    for(size_t row = 0; row < res->shape[ROWS]; ++row) {
        for(size_t col = 0; col < res->shape[COLS]; ++col) {
            double acc = 0.0;

            for(size_t k = 0; k < A->shape[COLS]; ++k) {
                acc += *Matrix_at(A, row, k) * *Matrix_at(B, k, col);
            }
            
            *Matrix_at(res, row, col) = acc;
        }
    }
    return res;
}

Matrix *Matrix_transpose(Matrix *A) {
    Matrix *T = malloc(sizeof(Matrix));
    size_t T_shape[2] = {A->shape[COLS], A->shape[ROWS]};
    Matrix_ctor(T, T_shape);
    for(size_t row = 0; row < T->shape[ROWS]; ++row) {
        for(size_t col = 0; col < T->shape[COLS]; ++col) {
            *Matrix_at(T, row, col) = *Matrix_at(A, col, row);
        }
    }

    return T;
}

Matrix *Matrix_ones_like(Matrix *this) {
    Matrix *ones = malloc(sizeof(*this));
    Matrix_ctor(ones, this->shape);
    for(size_t i = 0; i < ones->shape[0] * ones->shape[1]; ++i) {
        ones->val[i] = 1.0;
    }
    return ones;
}

void Matrix_to_ones(Matrix *this) {
    for(size_t i = 0; i < this->shape[0] * this->shape[1]; ++i) {
        this->val[i] = 1.0;
    }
}

Matrix *Matrix_zeroes(Matrix *this) {
    Matrix *zeroes = malloc(sizeof(*this));
    Matrix_ctor(zeroes, this->shape);
    for(size_t i = 0; i < zeroes->shape[0] * zeroes->shape[1]; ++i) {
        zeroes->val[i] = 0.0;
    }
    return zeroes;
}

void Matrix_mul_scalar_inplace(Matrix *matrix, double scalar) {
    for(size_t i = 0; i < matrix->shape[0] * matrix->shape[1]; ++i) {
        matrix->val[i] *= scalar;
    }
}

typedef struct MatrixOperation_ {
    void*(*call)(void*);
    void **args;
    size_t args_len;
    void*(*backward)(void*);
} MatrixOperation;


struct Connection_;
typedef struct RawFunc_ {
    void*(*call)(void*);
} RawFunc;
typedef enum LayerKind_ {
    NIL = 0,
    OP = (1 << 1),
    SYNAPSE = (1 << 2)
} LayerKind;
typedef struct Layer_ {
    LayerKind kind;
    union {
        MatrixOperation *op;
        struct Connection_ *synapse;
        RawFunc *raw;
    } as;
} Layer;

typedef struct {
    Layer this;
} MatrixOperation_arg0;
typedef struct {
    Layer this;
    Matrix *arg1;
} MatrixOperation_arg1;
typedef struct {
    Layer this;
    Matrix *arg1;
    Matrix *arg2;
} MatrixOperation_arg2;

MatrixOperation MatrixOperation_ctor() {
    MatrixOperation op = {0};
    return op;
}
void MatrixOperation_dtor(MatrixOperation op) {
    free(op.args);
}

void MatrixOperation_zero_grad(MatrixOperation *this) {
    for(size_t i = 0; i < this->args_len; ++i) {
        Matrix_zero_grad(((Matrix**)this->args)[i]);
    }
}

typedef struct Plus_ {
    MatrixOperation base;
} Plus;

void Plus_dtor(Plus adder) {
    MatrixOperation_dtor(adder.base);
}

void *Plus_call(void *args_) {
    MatrixOperation_arg2 *args = args_;
    Plus *this = (Plus*)args->this.as.op;
    Matrix *A = args->arg1;
    Matrix *B = args->arg1;

    Matrix *C = malloc(sizeof(Matrix));
    Matrix_ctor(C, A->shape);
    for(size_t i = 0; i < C->shape[0] * C->shape[1]; ++i) {
        C->val[i] = A->val[i] + B->val[i];
    }
    C->creator = this;

    this->base.args_len = 2;
    this->base.args = malloc(this->base.args_len * sizeof(void*));
    ((Matrix**)this->base.args)[0] = A;
    ((Matrix**)this->base.args)[1] = B;

    return C;
}

struct Plus_backward_args_t {
    Plus *this;
    void *s;
};
void *Plus_backward(void *args_) {
    struct Plus_backward_args_t *args = args_;
    Plus *this = args->this;
    void *s = args->s;

    Matrix *ones = Matrix_ones_like((Matrix*)s);
    Matrix *mat_of_s = Matrix_mul((double*)s, ones);
    Matrix_backward(((Matrix**)this->base.args)[0],  mat_of_s);
    Matrix_backward(((Matrix**)this->base.args)[1],  mat_of_s);

    Matrix_dtor(*ones);
    free(ones);
    Matrix_dtor(*mat_of_s);
    free(mat_of_s);

    return NULL;
}

Plus Plus_ctor() {
    Plus adder = {0};
    adder.base = MatrixOperation_ctor();
    adder.base.call = Plus_call;
    adder.base.backward = Plus_backward;
    return adder;
}

typedef struct Minus_ {
    MatrixOperation base;
} Minus;

void Minus_dtor(Minus adder) {
    MatrixOperation_dtor(adder.base);
}

void *Minus_call(void *args_) {
    MatrixOperation_arg2 *args = args_;
    Minus *this = (Minus*)args->this.as.op;
    Matrix *A = args->arg1;
    Matrix *B = args->arg1;

    Matrix *C = malloc(sizeof(Matrix));
    Matrix_ctor(C, A->shape);
    for(size_t i = 0; i < C->shape[0] * C->shape[1]; ++i) {
        C->val[i] = A->val[i] + B->val[i];
    }
    C->creator = this;

    this->base.args_len = 2;
    this->base.args = malloc(this->base.args_len * sizeof(void*));
    ((Matrix**)this->base.args)[0] = A;
    ((Matrix**)this->base.args)[1] = B;

    return (void*)C;
}

struct Minus_backward_args_t {
    Minus *this;
    void *s;
};
void *Minus_backward(void *args_) {
    struct Minus_backward_args_t *args = args_;
    Minus *this = args->this;
    void *s = args->s;

    Matrix *ones = Matrix_ones_like((Matrix*)s);
    Matrix_backward(((Matrix**)this->base.args)[0],  Matrix_mul((double*)s, ones));
    Matrix_mul_scalar_inplace(ones, -1.0);
    Matrix_backward(((Matrix**)this->base.args)[1],  Matrix_mul((double*)s, ones));
    Matrix_dtor(*ones);
    free(ones);

    return NULL;
}

Minus Minus_ctor() {
    Minus adder = {0};
    adder.base = MatrixOperation_ctor();
    adder.base.call = Minus_call;
    adder.base.backward = Minus_backward;
    return adder;
}

typedef struct Logistic_ NOTE(: MatrixOperation) {
    MatrixOperation base;
} Logistic;

double logistic(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double logistic_derivitive(double x) {
    return logistic(x) * (1.0 - logistic(x));
}

void *Logistic_call(void *args_) {
    MatrixOperation_arg1 *args = args_;
    Logistic *this = (Logistic*)args->this.as.op;
    Matrix *x = args->arg1;

    this->base.args_len = 1;
    this->base.args = realloc(this->base.args, this->base.args_len * sizeof(void*));
    ((Matrix**)this->base.args)[0] = x;
    Matrix *v = malloc(sizeof(Matrix));
    Matrix_ctor(v, x->shape);
    Matrix_apply(v, ((Matrix**)this->base.args)[0], logistic);
    v->creator = this;
    return v;
}

struct Logistic_backward_args_t {
    Logistic *this;
    Matrix *s;
};
void *Logistic_backward(void *args_) {
    struct Logistic_backward_args_t *args = args_;
    Logistic *this = args->this;
    Matrix *s = args->s;

    Matrix *der = malloc(sizeof(Matrix));
    Matrix_ctor(der, s->shape);
    Matrix_apply(der, ((Matrix**)this->base.args)[0], logistic_derivitive);

    Matrix *res = Matrix_product(s, der);
    Matrix_backward(((Matrix**)this->base.args)[0], res); 
    
    Matrix_dtor(*der);
    free(der);
    Matrix_dtor(*res);
    free(res);

    return NULL;
}

Logistic Logistic_ctor() {
    Logistic logi = {0};
    logi.base = MatrixOperation_ctor();
    logi.base.call = Logistic_call;
    logi.base.backward = Logistic_backward;
    return logi;
}

typedef struct Mul_ {
    MatrixOperation base;
} Mul;

void *Mul_call(void *args_) {
    MatrixOperation_arg2 *args = args_;
    Mul *this = (Mul*)args->this.as.op;
    Matrix *A = args->arg1;
    Matrix *B = args->arg2;

    this->base.args_len = 2;
    this->base.args = realloc(this->base.args, this->base.args_len * sizeof(Matrix));
    ((Matrix**)this->base.args)[0] = A;
    ((Matrix**)this->base.args)[1] = B;
    Matrix *v = Matrix_product(A, B);
    v->creator = this;
    return v;
}


struct Mul_backward_args_t {
    Mul *this;
    Matrix *s;
};
void *Mul_backward(void *args_) {
    struct Mul_backward_args_t *args = args_;
    Mul *this = args->this;
    Matrix *s = args->s;

    Matrix *arg0T = Matrix_transpose(((Matrix**)this->base.args)[0]);
    Matrix *arg1T = Matrix_transpose(((Matrix**)this->base.args)[1]);
    Matrix *new_s_left = Matrix_product(s, arg1T); 
    Matrix *new_s_right = Matrix_product(arg0T, s); 
    Matrix_backward(((Matrix**)this->base.args)[0], new_s_left);
    Matrix_backward(((Matrix**)this->base.args)[1], new_s_right);

    Matrix_dtor(*arg0T);
    Matrix_dtor(*arg1T);
    Matrix_dtor(*new_s_left);
    Matrix_dtor(*new_s_right);
    free(arg0T);
    free(arg1T);
    free(new_s_left);
    free(new_s_right);

    return NULL;
}

Mul Mul_ctor() {
    Mul mul = {0};
    mul.base = MatrixOperation_ctor();
    mul.base.call = Mul_call;
    mul.base.backward = Mul_backward;
    return mul;
}

void Mul_dtor(Mul mul) {
    MatrixOperation_dtor(mul.base);
}

typedef struct BinaryCE_ {
    MatrixOperation base;
    Matrix *target;
} BinaryCE;

Matrix *log_loss(Matrix *Y, Matrix *target) {
    Matrix *acc = malloc(sizeof(Matrix));
    Matrix_ctor(acc, Y->shape);
    Matrix_zeroes(acc);
    TODO("log loss");
    return NULL;
}

void *BinaryCE_call(void *args_) {
    MatrixOperation_arg2 *args = args_;
    BinaryCE *this = (BinaryCE*)args->this.as.op;
    Matrix *Y = args->arg1;
    Matrix *target = args->arg1;

    ((Matrix**)this->base.args)[0] = Y;
    this->target = target;
    Matrix *v = log_loss(Y, target);
    v->creator = this;
    return v;
}

struct BinaryCE_backward_args_t {
    BinaryCE *this;
    Matrix *s;
};
void *BinaryCE_backward(void *args_) {
    struct BinaryCE_backward_args_t *args = args_;
    BinaryCE *this = args->this;
    Matrix *s = args->s;

    Matrix *y = ((Matrix**)this->base.args)[0];
    double N = (double)y->shape[0];
    Matrix *t = this->target;
    Matrix *derivitive = malloc(sizeof(Matrix));
    Matrix_ctor(derivitive, y->shape);
    for(size_t i = 0; i < derivitive->shape[ROWS] * derivitive->shape[COLS]; ++i) {
        derivitive->val[i] = -(1.0/N) * ((t->val[i] / y->val[i]) * ((1.0 - t->val[i])/(1.0 - y->val[i])));
    }
    Matrix *next_s =  Matrix_product(s, derivitive);
    Matrix_backward(y, next_s);

    Matrix_dtor(*derivitive);
    Matrix_dtor(*next_s);
    free(derivitive);
    free(next_s);

    return NULL;
}

BinaryCE BinaryCE_ctor() {
    BinaryCE loss = {0};
    loss.base = MatrixOperation_ctor();
    loss.base.call = BinaryCE_call;
    loss.base.backward = BinaryCE_backward;
    loss.target = NULL;
    return loss;
}

typedef struct Connection_ {
    void *(*call)(void*);
    size_t in_size;
    size_t out_size;
    Matrix *weights;
    Matrix *bias;
    Matrix *params[2];
} Connection;
struct Connection_call_args_t {
    Connection *this;
    Matrix *x;
};

void *Connection_call(void *args_) {
    struct Connection_call_args_t *args = args_;
    Connection *this = args->this;
    Matrix *x = args->x;

    Mul *muller = malloc(sizeof(Mul));
    *muller = Mul_ctor();

    Plus *adder = malloc(sizeof(Plus));
    *adder = Plus_ctor();

    Matrix *h = (*muller->base.call)((void*)&(MatrixOperation_arg2){
        (MatrixOperation*)muller, x, this->weights
    });
    h->creator = muller;

    Matrix *ones = malloc(sizeof(Matrix));
    size_t ones_shape[2] = {h->shape[0], 1};
    Matrix_ctor(ones, ones_shape);
    Matrix_to_ones(ones);
    Matrix *big_bias = Matrix_product(ones, this->weights);
    
    Matrix *v = (*adder->base.call)((void*)&(MatrixOperation_arg2){
        (MatrixOperation*)adder, h, big_bias
    });
    v->creator = (void*)adder;
    
    Matrix_dtor(*ones);
    free(ones);

    return v;
}

Connection Connection_ctor(size_t from_nodes, size_t to_nodes) {
    Connection synapse;
    synapse.call = Connection_call;

    synapse.in_size = from_nodes;
    synapse.out_size = to_nodes;

    synapse.weights = malloc(sizeof(Matrix));
    Matrix_ctor(synapse.weights, (size_t*)(&synapse));
    Matrix_rand(synapse.weights);
    Matrix_mul_scalar_inplace(synapse.weights, 1.0 / sqrt((double)synapse.in_size));

    synapse.bias = malloc(sizeof(Matrix));
    size_t bias_shape[2] = {synapse.out_size, 1}; // vertical vector
    Matrix_ctor(synapse.bias, bias_shape);
    Matrix_zeroes(synapse.bias);

    return synapse;
}

void Connection_dtor(Connection synapse) {
    Matrix_dtor(*synapse.weights);
    free(synapse.weights);
    Matrix_dtor(*synapse.bias);
    free(synapse.bias);
}

typedef struct Dataset_ {
    Matrix *targets;
    size_t n;
} Dataset;

Matrix *Dataset_inputs(Dataset *const this) {
    return NULL;
}

Matrix *Dataset_targets(Dataset *const this) {
    return NULL;
}

Dataset Dataset_ctor() {
    Dataset set;
    return set;
}
typedef struct Network_ {
    size_t layers_capacity;
    size_t layers_len;
    Layer *layers;
    MatrixOperation *loss;
    size_t loss_history_capacity;
    size_t loss_history_len;
    double *loss_history;
    Matrix*(*call)(struct Network_*,Matrix*);
} Network;

// eats x
Matrix *Network_call(Network *this, Matrix *x) {
    for(Layer *layer = this->layers; layer != this->layers + this->layers_len; ++layer) {
        Matrix *new_x = (Matrix*)(*layer->as.raw->call)((void*)&(MatrixOperation_arg1){
            .this = *layer,
            .arg1 = x
        });

        Matrix_dtor(*x);
        free(x);
        x = new_x;
    }
    return x;
}

Network Network_ctor() {
    Network net;
    net.call = Network_call;
    net.layers_capacity = 16;
    net.layers_len = 0;
    net.layers = calloc(net.layers_capacity, sizeof(*net.layers));
    net.loss = NULL;
    net.loss_history_capacity = 16;
    net.loss_history_len = 0;
    net.loss_history = calloc(net.loss_history_capacity, sizeof(*net.loss_history));
    return net;
}

void Network_append_layer(Network *this, Layer layer) {
    if(this->layers_len == this->layers_capacity) {
        this->layers_capacity *= 2;
        this->layers = realloc(this->layers, this->layers_capacity * sizeof(*this->layers));
    }
    this->layers[this->layers_len++] = layer;
}

void Network_append_loss_history(Network *this, double loss) {
    if(this->loss_history_len == this->loss_history_capacity) {
        this->loss_history_capacity *= 2;
        this->loss_history = realloc(this->loss_history, this->loss_history_capacity * sizeof(*this->loss_history));
    }
    this->loss_history[this->loss_history_len++] = loss;
}

void Network_learn(Network *this, Dataset *data, double learning_rate, size_t epochs) {
    Matrix *x = Dataset_inputs(data);
    Matrix *target = Dataset_targets(data);
    for(size_t epoch = 0; epoch < epochs; ++epoch) {
        Matrix *y = (*this->call)(this, x);
        Matrix *loss = (*this->loss->call)((void*)&(MatrixOperation_arg2){ this->loss, y, target });
        MatrixOperation_zero_grad(this->loss);

        {
            Matrix one;
            size_t singleton[2] = {1,1};
            Matrix_ctor(&one, singleton);
            one.val[0] = 1.0;
            (*this->loss->backward)(&one);
        }

        // for(Matrix **param = this->params; param != this->params + 2; ++p) {
        //     for(size_t i = 0; i < (*param)->shape[0] * (*param)->shape[1]; ++i) {
        //         (*param)->val[i] -= learning_rate * (*param)->grad[i];
        //     }
        // }
        if(epoch % 50 == 0) {
            printf("%zu: cost = %f", epoch, *Matrix_at(loss, 0, 0));
        }
    }
}

int main(void) {
    srand(time(NULL));
    Matrix A;
    size_t shape[2] = {3,3};
    Matrix_ctor(&A, shape);
    *Matrix_at(&A, randi(0,3), randi(0,3)) = randf_normal();
    *Matrix_at(&A, randi(0,3), randi(0,3)) = randf_normal();
    *Matrix_at(&A, randi(0,3), randi(0,3)) = randf_normal();
    Matrix_apply(&A, &A, exp);

    Matrix B;
    Matrix_ctor(&B, shape);
    Matrix_to_ones(&B);

    printf("A == \n"); Matrix_cout(A);
    printf("b == \n"); Matrix_cout(B);

    Matrix *C = Matrix_product(&A,&B);
    printf("C == \n"); Matrix_cout(*C);

    Matrix_dtor(A);
    Matrix_dtor(B);

    Matrix_dtor(*C);
    free(C);

    return 0;

    srand(time(NULL));
    Dataset *my_dataset = malloc(sizeof(Dataset));
    *my_dataset = Dataset_ctor();
    //Dataset_plot(my_dataset, NULL);

    return 0;

    Network *net = malloc(sizeof(Network));
    *net = Network_ctor();
    
    {
        Layer layer;

        layer.as.synapse = malloc(sizeof(Connection));
        *layer.as.synapse = Connection_ctor(2, 50);
        Network_append_layer(net, layer);

        layer.as.op = malloc(sizeof(Logistic));
        *(Logistic*)layer.as.op = Logistic_ctor();
        Network_append_layer(net, layer);

        layer.as.synapse = malloc(sizeof(Connection));
        *layer.as.synapse = Connection_ctor(50, 50);
        Network_append_layer(net, layer);

        layer.as.op = malloc(sizeof(Logistic));
        *(Logistic*)layer.as.op = Logistic_ctor();
        Network_append_layer(net, layer);

        layer.as.synapse = malloc(sizeof(Connection));
        *layer.as.synapse = Connection_ctor(50, 1);
        Network_append_layer(net, layer);

        layer.as.op = malloc(sizeof(Logistic));
        *(Logistic*)layer.as.op = Logistic_ctor();
        Network_append_layer(net, layer);
    }
    BinaryCE *loss = malloc(sizeof(BinaryCE));
    *loss = BinaryCE_ctor();
    net->loss = (MatrixOperation*)loss;

    // learn
    Network_learn(net, my_dataset, 1, 7000);
    //plot_vector(net->loss_history, net->loss_history_len, "Epoch", "Loss"); 

    // test
    Matrix *inputs = Dataset_inputs(my_dataset);
    Matrix *y = (*net->call)(net, inputs);
    //Dataset_plot(my_dataset, y);

    Matrix_dtor(*inputs);
    Matrix_dtor(*y);
    free(inputs);
    free(y);

    return 0;
}
