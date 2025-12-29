/**
 * Project: fastLM Engine
 * File: src/main.cpp
 * Description: Transformer Architecture with Numerical Stability Fixes
 */

#include <iostream>
#include <vector>
#include <cmath>     // exp, sqrt
#include <iomanip>   // setprecision
#include <cstdlib>   // rand

typedef std::vector<std::vector<float>> Matrix;

// ------------------------------------------------------------------
// HELPER: Print Matrix
// ------------------------------------------------------------------
void printMatrix(const std::string& label, const Matrix& m) {
    std::cout << "--- " << label << " [" << m.size() << "x" << m[0].size() << "] ---" << std::endl;
    for (const auto& row : m) {
        std::cout << "[ ";
        for (float val : row) {
            std::cout << std::fixed << std::setprecision(4) << val << " "; // 4 decimal places
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
}

// ------------------------------------------------------------------
// 🔧 FIXED: Random Initialization
// Old version: rand() % 10  (0 to 9) -> CAUSED NAN EXPLOSION
// New version: (float)rand() / RAND_MAX (0.0 to 1.0) -> STABLE
// ------------------------------------------------------------------
Matrix createRandomMatrix(int rows, int cols) {
    Matrix m(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Generates a float between 0.0 and 1.0
            m[i][j] = (float)rand() / RAND_MAX; 
        }
    }
    return m;
}

// ------------------------------------------------------------------
// MATH OPERATIONS
// ------------------------------------------------------------------
Matrix matmul(const Matrix& A, const Matrix& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) {
        std::cerr << "Dimension mismatch!" << std::endl;
        exit(1);
    }

    Matrix C(rowsA, std::vector<float>(colsB, 0.0f));
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

Matrix transpose(const Matrix& m) {
    int rows = m.size();
    int cols = m[0].size();
    Matrix result(cols, std::vector<float>(rows)); 
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j][i] = m[i][j];
        }
    }
    return result;
}

void softmax(Matrix& m) {
    int rows = m.size();
    int cols = m[0].size();
    for (int i = 0; i < rows; i++) {
        float sum_exp = 0.0f;
        
        // Stability Fix: Find max value in row to prevent overflow
        // (This is called the "Log-Sum-Exp" trick)
        float max_val = -1e9;
        for(float val : m[i]) if(val > max_val) max_val = val;

        for (int j = 0; j < cols; j++) {
            // Subtract max_val for numerical stability
            m[i][j] = std::exp(m[i][j] - max_val); 
            sum_exp += m[i][j];
        }
        
        for (int j = 0; j < cols; j++) {
            m[i][j] /= sum_exp;
        }
    }
}

Matrix attention(const Matrix& Q, const Matrix& K, const Matrix& V) {
    Matrix K_T = transpose(K);
    Matrix scores = matmul(Q, K_T);

    float d_k = (float)Q[0].size();
    float scale = std::sqrt(d_k);

    for (int i = 0; i < scores.size(); i++) {
        for (int j = 0; j < scores[0].size(); j++) {
            scores[i][j] /= scale;
        }
    }

    softmax(scores);
    return matmul(scores, V);
}

// ------------------------------------------------------------------
// 🛠️ HELPER: Load Matrix from File
// Reads 'rows * cols' floats from the open file stream
// ------------------------------------------------------------------
void loadMatrix(FILE* f, Matrix& m) {
    int rows = m.size();
    int cols = m[0].size();
    for (int i = 0; i < rows; i++) {
        // Read directly into the vector's memory
        // .data() gives us the pointer to the raw array inside the vector
        fread(m[i].data(), sizeof(float), cols, f);
    }
}

struct TransformerBlock {
    Matrix W_q, W_k, W_v, W_out;

    // Constructor NOW takes a file pointer!
    TransformerBlock(int d_model, FILE* f) {
        // Initialize empty matrices
        W_q = Matrix(d_model, std::vector<float>(d_model));
        W_k = Matrix(d_model, std::vector<float>(d_model));
        W_v = Matrix(d_model, std::vector<float>(d_model));
        W_out = Matrix(d_model, std::vector<float>(d_model));

        // Load data from file directly into these matrices
        std::cout << "  > Loading W_q..." << std::endl;
        loadMatrix(f, W_q);
        
        std::cout << "  > Loading W_k..." << std::endl;
        loadMatrix(f, W_k);
        
        std::cout << "  > Loading W_v..." << std::endl;
        loadMatrix(f, W_v);
        
        std::cout << "  > Loading W_out..." << std::endl;
        loadMatrix(f, W_out);
    }

    Matrix forward(const Matrix& input) {
        Matrix Q = matmul(input, W_q);
        Matrix K = matmul(input, W_k);
        Matrix V = matmul(input, W_v);
        Matrix attn_out = attention(Q, K, V);
        return matmul(attn_out, W_out);
    }
};

int main() {
    std::cout << "🚀 fastLM Engine v0.4 (Model Loader)..." << std::endl;
    
    // 1. Open the Model File
    FILE* f = fopen("models/model.bin", "rb"); // rb = Read Binary
    if (!f) {
        std::cerr << "❌ Error: Could not open models/model.bin" << std::endl;
        return 1;
    }

    // 2. Read Header
    unsigned int magic;
    fread(&magic, sizeof(int), 1, f);
    if (magic != 0xFEEDBEEF) {
        std::cerr << "❌ Error: Invalid file format!" << std::endl;
        return 1;
    }
    std::cout << "✅ File verified (Magic: " << std::hex << magic << ")" << std::endl;

    // 3. Read Dimensions
    int layers, d_model;
    fread(&layers, sizeof(int), 1, f);
    fread(&d_model, sizeof(int), 1, f);
    std::cout << "  Model Config: Layers=" << std::dec << layers 
              << ", d_model=" << d_model << std::endl;

    // 4. Load the Transformer Block
    // We pass the file pointer 'f' so the block can read its own weights
    TransformerBlock layer1(d_model, f);
    
    // Close file (we are done reading)
    fclose(f);

   // ... inside main(), after loading the model ...

    // 5. Run Inference (With Timer)
    std::cout << "\n🧠 Running Inference with Loaded Weights..." << std::endl;
    
    // Create Dummy Input
    Matrix input(3, std::vector<float>(d_model));
    for(int i=0; i<3; i++) for(int j=0; j<d_model; j++) input[i][j] = 0.5f;

    // --- START TIMER ---
    auto start = std::chrono::high_resolution_clock::now();

    // The heavy work
    Matrix output = layer1.forward(input);

    // --- STOP TIMER ---
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration in microseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printMatrix("Final Output", output);

    std::cout << "⚡ Inference Time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "🚀 Speed: " << (1000000.0 / duration.count()) << " tokens/second (approx)" << std::endl;

    return 0;

}