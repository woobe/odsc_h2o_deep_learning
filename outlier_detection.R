# ------------------------------------------------------------------------------------------------
# Outlier Detection
# Based on Boston Housing Data (same as step_05_regression_basic.R)

# Also see https://github.com/h2oai/h2o-training-book/blob/master/hands-on_training/anomaly_detection.md
# for MNIST outlier detection
# ------------------------------------------------------------------------------------------------

# Load R Packages
suppressPackageStartupMessages(library(h2o))
suppressPackageStartupMessages(library(mlbench)) # for datasets

# Start and connect to a local H2O cluster
h2o.init(nthreads = -1)
h2o.no_progress()

# Load BostonHousing dataset from mlbench
data(BostonHousing)

# Use as.h2o() to convert R data frame into H2O data frame
h_boston <- as.h2o(BostonHousing)

# Quick summary
h2o.describe(h_boston)

# Define target (y) and features (x)
target <- "medv" # median house value
features <- setdiff(colnames(h_boston), target)
print(features)


# ------------------------------------------------------------------------------------------------
# Visualise dataset in 2D using principle components
# ------------------------------------------------------------------------------------------------

# Run a Principal Components Analysis (PCA)
model_pca <- h2o.prcomp(training_frame = h_boston, 
                        x = features, 
                        k = 2,
                        transform = "STANDARDIZE",
                        pca_method = "GLRM",
                        use_all_factor_levels = TRUE,
                        seed = 1234)

# Extract the first two principle components
h_pca <- h2o.predict(model_pca, h_boston)

# Visualise
d_pca <- as.data.frame(h_pca)
plot(d_pca, main = "First Two Principle Components of Boston Housing Data")


# ------------------------------------------------------------------------------------------------
# Train a deep autoencoder and visualise reconstruction errors
# ------------------------------------------------------------------------------------------------

# Training a Deep Autoencoder
model <- h2o.deeplearning(x = features,
                          training_frame = h_boston,
                          autoencoder = TRUE,
                          activation = "Tanh",
                          hidden = c(100, 100, 100),
                          epochs = 100,
                          seed = 1234,
                          reproducible = TRUE)

# Calculate reconstruction errors (MSE)
recon_errors <- h2o.anomaly(model, h_boston, per_feature = FALSE)
print(recon_errors)

# Convert H2O data frame into R data types
d_errors <- as.data.frame(recon_errors)
n_errors <- as.numeric(recon_errors)

# user-defined cut-off point
cutoff <- quantile(n_errors, probs = 0.95) 

# Identify Outliers
row_outliers <- which(d_errors > cutoff) # based on plot above

# Plot
plot(sort(d_errors$Reconstruction.MSE), main = "Reconstruction Error")
abline(h = cutoff, col = "red") # red line = cutoff point

# Print outliers
print(BostonHousing[row_outliers, ])

# Visualise
d_pca$outlier <- 0
d_pca[row_outliers, ]$outlier <- 1
plot(d_pca[, 1:2], col = as.factor(d_pca$outlier),
     main = "First Two Principle Components of Boston Housing Data\n
             with Outliers Highlighted in Red")

