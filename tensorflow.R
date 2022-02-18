---
title: "Predicting LES in Congress with Neural Network"
output: html_document
---

# load some libraries
library(keras)
library(tensorflow)
library(tidyverse)
library(patchwork)
library(skimr)
library(naniar)
library(recipes)
library(here)
library(tictoc)
library(dplyr)

#install_keras()
#install_tensorflow()

# load the data
cong <- read_csv(here("congress_109_110.csv"))

data <- cong %>% 
  select(elected, votepct, dwnom1, seniority, les) %>% 
  glimpse()

data %>%
  gg_miss_upset()

# impute
congress_recipe <- recipe(les ~ elected + votepct + dwnom1 + seniority, 
                          data = data) %>%
  step_meanimpute(dwnom1) %>% 
  step_knnimpute(all_predictors()) 

congress_imputed <- prep(congress_recipe) %>% 
  juice()

# glimpse
summary(data$dwnom1)
summary(congress_imputed$dwnom1) 

summary(data$elected)
summary(congress_imputed$elected) 

summary(data$seniority)
summary(congress_imputed$seniority) 

summary(data$votepct)
summary(congress_imputed$votepct)


set.seed(1234)

## first, create train test split (80/20)
congress_imputed_split <- sample(1:nrow(congress_imputed), 0.8 * nrow(congress_imputed))

train <- congress_imputed[congress_imputed_split, ]
test <- congress_imputed[-congress_imputed_split, ]

train <- train %>% 
  scale() 

train_mean <- attr(train, "scaled:center") 
train_sd <- attr(train, "scaled:scale")
test <- scale(test, center = train_mean, scale = train_sd)

train_labels = train[ , "les"]
train_data = train[ , 1:4]

test_labels = test[ , "les"]
test_data = test[ , 1:4]

# initialize model
model <- keras_model_sequential() 

model %>% 
      layer_dense(units = 8,
                activation = "relu",
                input_shape = dim(train_data)[2]) %>%
      layer_dense(units = 8,
                activation = "relu") %>%
      layer_dense(units = 8,
                activation = "relu") %>%
    layer_dense(units = 1) %>% 
  compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list("mae")
  )

epochs <- 500

# fit (predicting LES)
tic()
out <- model %>% fit(
  train_data,
  train_labels,
  epochs = epochs,
  validation_split = 0.2,
  verbose = FALSE
); out 
toc() 

# viz
plot(out, loss = "mse") +
  theme_minimal() +
  labs(title = "Predicting LES in Congress with Neural Network",
       subtitle = "Three Dense 8-Neuron Layer + ReLU Activation",
       x = "Epoch",
       y = "Metric")

# preds
test_predictions <- model %>% 
  predict(test_data)

# compare with raw LES
preds <- qplot(test_predictions, xlab = "Predicted LES")
peak_preds <- ggplot_build(preds)[[1]][[1]]
x1 <- mean(unlist(peak_preds[which.max(peak_preds$ymax), c("xmin", "xmax")])); x1
preds <- preds + 
  geom_vline(xintercept=x1, col="red", lty=2, lwd=1) + 
  geom_vline(xintercept=0, col="blue", lty=1, lwd=1) + 
  theme_minimal()

trained <- qplot(train_labels, xlab = "Training LES")
peak_tr <- ggplot_build(trained)[[1]][[1]]
x2 <- mean(unlist(peak_tr[which.max(peak_tr$ymax), c("xmin", "xmax")])); x2
trained <- trained + 
  geom_vline(xintercept=x2, col="red", lty=2, lwd=1) + 
  geom_vline(xintercept=0, col="blue", lty=1, lwd=1) + 
  theme_minimal()

# viz side by side
preds + 
  trained + plot_annotation(title = "Comparing Predicted and Raw LES")


