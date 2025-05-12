#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <time.h>

#define IMAGE_SIZE (28 * 28)
#define NUM_CLASSES 10
#define NUM_HIDDEN_1 512
#define NUM_HIDDEN_2 256
#define INITIAL_LEARNING_RATE 0.001
#define EPOCHS 100
#define TRAIN_SAMPLES 60000
#define TEST_SAMPLES 10000
#define TRAIN_IMAGES_PATH "train-images.idx3-ubyte"
#define TRAIN_LABELS_PATH "train-labels.idx1-ubyte"
#define TEST_IMAGES_PATH "t10k-images.idx3-ubyte"
#define TEST_LABELS_PATH "t10k-labels.idx1-ubyte"
#define MODEL_PATH "10k.bin"

typedef struct {
    int index;
    float confidence;
} TopPrediction;

float W1[NUM_HIDDEN_1][IMAGE_SIZE];
float B1[NUM_HIDDEN_1];
float W2[NUM_HIDDEN_2][NUM_HIDDEN_1];
float B2[NUM_HIDDEN_2];
float W3[NUM_CLASSES][NUM_HIDDEN_2];
float B3[NUM_CLASSES];

float train_images[TRAIN_SAMPLES][IMAGE_SIZE];
int train_labels[TRAIN_SAMPLES];
float test_images[TEST_SAMPLES][IMAGE_SIZE];
int test_labels[TEST_SAMPLES];

void relu(float *input, int size) {
    for (int i = 0; i < size; i++) {
        if (input[i] < 0) input[i] = 0;
    }
}

float relu_derivative(float x) {
    return (x > 0) ? 1.0 : 0.0;
}

void softmax(float *input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    float sum = 0;
    for (int i = 0; i < size; i++) {
        input[i] = exp(input[i] - max_val);
        sum += input[i];
    }
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

void initialize_weights() {
    srand(time(NULL));
    for (int i = 0; i < NUM_HIDDEN_1; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            W1[i][j] = ((float) rand() / RAND_MAX - 0.5) * sqrt(2.0 / IMAGE_SIZE);
        }
        B1[i] = 0.01;
    }
    for (int i = 0; i < NUM_HIDDEN_2; i++) {
        for (int j = 0; j < NUM_HIDDEN_1; j++) {
            W2[i][j] = ((float) rand() / RAND_MAX - 0.5) * sqrt(2.0 / NUM_HIDDEN_1);
        }
        B2[i] = 0.01;
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < NUM_HIDDEN_2; j++) {
            W3[i][j] = ((float) rand() / RAND_MAX - 0.5) * sqrt(2.0 / NUM_HIDDEN_2);
        }
        B3[i] = 0.0;
    }
}

void forward(float *image, float *hidden1, float *hidden2, float *output) {
    for (int i = 0; i < NUM_HIDDEN_1; i++) {
        hidden1[i] = B1[i];
        for (int j = 0; j < IMAGE_SIZE; j++) {
            hidden1[i] += W1[i][j] * image[j];
        }
    }
    relu(hidden1, NUM_HIDDEN_1);

    for (int i = 0; i < NUM_HIDDEN_2; i++) {
        hidden2[i] = B2[i];
        for (int j = 0; j < NUM_HIDDEN_1; j++) {
            hidden2[i] += W2[i][j] * hidden1[j];
        }
    }
    relu(hidden2, NUM_HIDDEN_2);

    for (int i = 0; i < NUM_CLASSES; i++) {
        output[i] = B3[i];
        for (int j = 0; j < NUM_HIDDEN_2; j++) {
            output[i] += W3[i][j] * hidden2[j];
        }
    }
    softmax(output, NUM_CLASSES);
}

void bubble_sort(TopPrediction *preds, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (preds[j].confidence < preds[j + 1].confidence) {
                TopPrediction temp = preds[j];
                preds[j] = preds[j + 1];
                preds[j + 1] = temp;
            }
        }
    }
}


void load_mnist_data(const char *image_path, const char *label_path, float images[][IMAGE_SIZE], int *labels, int num_samples) {
    FILE *image_file = fopen(image_path, "rb");
    FILE *label_file = fopen(label_path, "rb");
    if (!image_file || !label_file) {
        fprintf(stderr, "Error: Could not open MNIST data files.\n");
        exit(1);
    }

    fseek(image_file, 16, SEEK_SET);
    fseek(label_file, 8, SEEK_SET);

    for (int i = 0; i < num_samples; i++) {
        uint8_t label;
        fread(&label, 1, 1, label_file);
        labels[i] = label;

        for (int j = 0; j < IMAGE_SIZE; j++) {
            uint8_t pixel;
            fread(&pixel, 1, 1, image_file);
            images[i][j] = pixel / 255.0;
        }
    }

    fclose(image_file);
    fclose(label_file);
}

void save_model(const char *model_path) {
    FILE *f = fopen(model_path, "wb");
    if (!f) {
        fprintf(stderr, "Error saving model.\n");
        return;
    }
    fwrite(W1, sizeof(W1), 1, f);
    fwrite(B1, sizeof(B1), 1, f);
    fwrite(W2, sizeof(W2), 1, f);
    fwrite(B2, sizeof(B2), 1, f);
    fwrite(W3, sizeof(W3), 1, f);
    fwrite(B3, sizeof(B3), 1, f);
    fclose(f);
    printf("Model saved to %s\n", model_path);
}

void load_model(const char *model_path) {
    FILE *f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "Error loading model.\n");
        exit(1);
    }
    fread(W1, sizeof(float), NUM_HIDDEN_1 * IMAGE_SIZE, f);
    fread(B1, sizeof(float), NUM_HIDDEN_1, f);
    fread(W2, sizeof(float), NUM_HIDDEN_2 * NUM_HIDDEN_1, f);
    fread(B2, sizeof(float), NUM_HIDDEN_2, f);
    fread(W3, sizeof(float), NUM_CLASSES * NUM_HIDDEN_2, f);
    fread(B3, sizeof(float), NUM_CLASSES, f);
    fclose(f);
    printf("Model loaded from %s\n", model_path);
}

void load_custom_image(const char *filename, float *image) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error loading image: %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < IMAGE_SIZE; i++) {
        unsigned char pixel;
        fread(&pixel, 1, 1, f);
        image[i] = pixel / 255.0f;
    }
    fclose(f);
}

void predict_custom_image(float *image) {
    float hidden1[NUM_HIDDEN_1], hidden2[NUM_HIDDEN_2], output[NUM_CLASSES];
    forward(image, hidden1, hidden2, output);

    TopPrediction preds[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; i++) {
        preds[i].index = i;
        preds[i].confidence = output[i];
    }
    bubble_sort(preds, NUM_CLASSES); 


    printf("Top 3 Predictions:\n");
    for (int i = 0; i < 3; i++) {
        printf("%d: %.2f%%\n", preds[i].index, preds[i].confidence * 100.0);
    }
}

void evaluate(const char *image_path, const char *label_path, int num_samples) {
    int correct = 0;
    int true_positives[NUM_CLASSES] = {0};
    int false_positives[NUM_CLASSES] = {0};
    int false_negatives[NUM_CLASSES] = {0};
    int confusion_matrix[NUM_CLASSES][NUM_CLASSES] = {0};

    float hidden1[NUM_HIDDEN_1], hidden2[NUM_HIDDEN_2], output[NUM_CLASSES];
    float (*eval_images)[IMAGE_SIZE] = malloc(num_samples * IMAGE_SIZE * sizeof(float));
    int *eval_labels = malloc(num_samples * sizeof(int));

    load_mnist_data(image_path, label_path, eval_images, eval_labels, num_samples);

    FILE *fp = fopen("predictions.csv", "w");
    if (fp) {
        fprintf(fp, "Index,Actual,Predicted\n");
    }

    for (int i = 0; i < num_samples; i++) {
        forward(eval_images[i], hidden1, hidden2, output);

        int predicted = 0;
        for (int j = 1; j < NUM_CLASSES; j++) {
            if (output[j] > output[predicted]) predicted = j;
        }

        int actual = eval_labels[i];
        confusion_matrix[actual][predicted]++;

        if (fp) {
            fprintf(fp, "%d,%d,%d\n", i, actual, predicted);
        }

        if (predicted == actual) {
            correct++;
            true_positives[actual]++;
        } else {
            false_positives[predicted]++;
            false_negatives[actual]++;
        }
    }

    if (fp) fclose(fp);

    float accuracy = (float)correct / num_samples * 100.0;
    printf("\nEvaluation Accuracy: %.2f%%\n", accuracy);
    printf("Correct predictions: %d / %d\n\n", correct, num_samples);

    printf("Class-wise Precision, Recall, F1-score:\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        int tp = true_positives[i];
        int fp = false_positives[i];
        int fn = false_negatives[i];

        float precision = (tp + fp) ? (float)tp / (tp + fp) : 0;
        float recall = (tp + fn) ? (float)tp / (tp + fn) : 0;
        float f1 = (precision + recall) ? 2 * precision * recall / (precision + recall) : 0;

        printf("Class %d: Precision = %.2f, Recall = %.2f, F1-score = %.2f\n", i, precision, recall, f1);
    }

    printf("\nConfusion Matrix:\n     ");
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("%3d ", i);
    }
    printf("\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("%3d: ", i);
        for (int j = 0; j < NUM_CLASSES; j++) {
            printf("%3d ", confusion_matrix[i][j]);
        }
        printf("\n");
    }

    free(eval_images);
    free(eval_labels);
}


float get_learning_rate(int epoch) {
    return INITIAL_LEARNING_RATE / (1.0 + 0.05 * epoch);
}

void train(float learning_rate) {
    float hidden1[NUM_HIDDEN_1], hidden2[NUM_HIDDEN_2], output[NUM_CLASSES];
    float delta_output[NUM_CLASSES], delta_hidden2[NUM_HIDDEN_2], delta_hidden1[NUM_HIDDEN_1];

    for (int i = 0; i < TRAIN_SAMPLES; i++) {
        forward(train_images[i], hidden1, hidden2, output);

        for (int j = 0; j < NUM_CLASSES; j++) {
            delta_output[j] = (j == train_labels[i] ? 1.0 : 0.0) - output[j];
        }

        for (int j = 0; j < NUM_HIDDEN_2; j++) {
            float grad_w = 0;
            for (int k = 0; k < NUM_CLASSES; k++) {
                grad_w += delta_output[k] * hidden2[j];
            }
            for (int k = 0; k < NUM_CLASSES; k++) {
                W3[k][j] += learning_rate * grad_w;
            }
            delta_hidden2[j] = 0;
            for (int k = 0; k < NUM_CLASSES; k++) {
                delta_hidden2[j] += delta_output[k] * W3[k][j];
            }
        }

        for (int j = 0; j < NUM_CLASSES; j++) {
            B3[j] += learning_rate * delta_output[j];
        }

        for (int j = 0; j < NUM_HIDDEN_2; j++) {
            delta_hidden2[j] *= relu_derivative(hidden2[j]);
        }

        for (int j = 0; j < NUM_HIDDEN_1; j++) {
            float grad_w = 0;
            for (int k = 0; k < NUM_HIDDEN_2; k++) {
                grad_w += delta_hidden2[k] * hidden1[j];
            }
            for (int k = 0; k < NUM_HIDDEN_2; k++) {
                W2[k][j] += learning_rate * grad_w;
            }
            delta_hidden1[j] = 0;
            for (int k = 0; k < NUM_HIDDEN_2; k++) {
                delta_hidden1[j] += delta_hidden2[k] * W2[k][j];
            }
        }

        for (int j = 0; j < NUM_HIDDEN_2; j++) {
            B2[j] += learning_rate * delta_hidden2[j];
        }

        for (int j = 0; j < NUM_HIDDEN_1; j++) {
            delta_hidden1[j] *= relu_derivative(hidden1[j]);
        }

        for (int j = 0; j < NUM_HIDDEN_1; j++) {
            for (int k = 0; k < IMAGE_SIZE; k++) {
                W1[j][k] += learning_rate * delta_hidden1[j] * train_images[i][k];
            }
            B1[j] += learning_rate * delta_hidden1[j];
        }
    }
}

int file_exists(const char *filename) {
    struct stat buffer;
    return stat(filename, &buffer) == 0;
}

int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "predict") == 0) {
        if (argc != 3) {
            fprintf(stderr, "Usage: %s predict <image_path>\n", argv[0]);
            return 1;
        }

        const char *image_path = argv[2];
        if (!file_exists(image_path)) {
            fprintf(stderr, "Custom image file '%s' not found.\n", image_path);
            return 1;
        }

        load_model(MODEL_PATH);
        float custom_image[IMAGE_SIZE];
        load_custom_image(image_path, custom_image);
        predict_custom_image(custom_image);
        return 0;
    }

    if (file_exists(MODEL_PATH)) {
        printf("Loading pre-trained model...\n");
        load_model(MODEL_PATH);
    } else {
        printf("Initializing weights...\n");
        initialize_weights();

        printf("Loading MNIST training dataset...\n");
        load_mnist_data(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, train_images, train_labels, TRAIN_SAMPLES);

        printf("Training the model...\n");
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            float lr = get_learning_rate(epoch);
            train(lr);
            printf("Epoch %d completed with learning rate: %f\n", epoch + 1, lr);
        }

        save_model(MODEL_PATH);
    }

    printf("Evaluating the model on the test set...\n");
    evaluate(TEST_IMAGES_PATH, TEST_LABELS_PATH, TEST_SAMPLES);
    return 0;
}
