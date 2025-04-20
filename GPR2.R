# Load libraries
library(tidyverse)
library(keras)
library(tensorflow)
library(tibble)

#Seed for consistent results
set.seed(42)
tensorflow::tf$random$set_seed(42)

# Load dataset
load("data_ml.RData")
features_short <- c("Div_Yld", "Eps", "Mkt_Cap_12M_Usd", "Mom_11M_Usd", "Ocf", "Pb", "Vol1Y_Usd")
data <- data_ml %>%
  filter(!is.na(R1M_Usd)) %>%
  group_by(date) %>%
  mutate(across(all_of(features_short), scale)) %>%
  ungroup()

# Convert features and returns to tensors to use with TensorFlow
get_state_tensor <- function(df) {
  X <- df %>% select(all_of(features_short)) %>% as.matrix()
  tf$convert_to_tensor(X, dtype = "float32")
}

get_reward_tensor <- function(df) {
  y <- df$R1M_Usd
  tf$convert_to_tensor(as.matrix(y), dtype = "float32")
}

# Actor and Critic
build_actor <- function(input_dim) {
  model <- keras_model_sequential()
  model$add(layer_dense(units = 64, activation = "relu", input_shape = list(input_dim)))
  model$add(layer_dense(units = 32, activation = "relu"))
  model$add(layer_dense(units = 1, activation = "tanh"))
  return(model)
}

build_critic <- function(input_dim) {
  model <- keras_model_sequential()
  model$add(layer_dense(units = 64, activation = "relu", input_shape = list(input_dim)))
  model$add(layer_dense(units = 32, activation = "relu"))
  model$add(layer_dense(units = 1))
  return(model)
}

# input features
input_dim <- length(features_short)
# initialize NN
actor <- build_actor(input_dim)
critic <- build_critic(input_dim)
# weights
actor_optimizer <- optimizer_adam(learning_rate = 0.001)
critic_optimizer <- optimizer_adam(learning_rate = 0.002)

dates <- sort(unique(data$date))

# DRL Training Loop (set epochs, extract inputs and outputs, apply gradients)
train_drl <- function(data, epochs = 5) {
  for (epoch in 1:epochs) {
    for (t in 2:length(dates)) {
      df_prev <- data %>% filter(date == dates[t - 1])
      df_curr <- data %>% filter(date == dates[t])
      if (nrow(df_prev) < 1 || nrow(df_curr) < 1) next
      state <- get_state_tensor(df_prev)
      next_state <- get_state_tensor(df_curr)
      reward <- get_reward_tensor(df_curr)
      
      with(tf$GradientTape(persistent = TRUE) %as% tape, {
        action <- actor(state)
        value <- critic(state)
        next_value <- critic(next_state)
        target <- reward + 0.99 * next_value
        advantage <- target - value
        critic_loss <- tf$reduce_mean(advantage^2)
        actor_loss <- -tf$reduce_mean(action * tf$stop_gradient(advantage))
      })
      
      critic_grads <- tape$gradient(critic_loss, critic$trainable_variables)
      actor_grads <- tape$gradient(actor_loss, actor$trainable_variables)
      
      critic_optimizer$apply_gradients(purrr::transpose(list(critic_grads, critic$trainable_variables)))
      actor_optimizer$apply_gradients(purrr::transpose(list(actor_grads, actor$trainable_variables)))
    }
  }
  return(list(actor = actor, critic = critic))
}

stock_ids <- levels(as.factor(data_ml$stock_id)) # converts stock id into factor

stock_days <- data_ml %>%
  group_by(stock_id) %>%
  summarize(nb = n(), .groups = "drop")

stock_ids_short <- stock_ids[which(stock_days$nb == max(stock_days$nb))] # identify stock with most obervations

# Train DRL (will take some time)
drl_data <- data %>% filter(stock_id %in% stock_ids_short)
trained_drl <- train_drl(drl_data, epochs = 5)

# Small and Large Neural Networks

# Generate simple x, y = sin(x)
x <- seq(0, 6, by = 0.001)
y <- sin(x)
x_matrix <- matrix(x, ncol = 1)
y_matrix <- matrix(y, ncol = 1)

# Generate high-resolution data
x_high <- seq(0, 6, by = 0.0002)
y_high <- sin(x_high)

x_high_matrix <- matrix(x_high, ncol = 1)
y_high_matrix <- matrix(y_high, ncol = 1)

# Convert to float32 tensors
x_high_tensor <- tf$convert_to_tensor(x_high_matrix, dtype = "float32")
y_high_tensor <- tf$convert_to_tensor(y_high_matrix, dtype = "float32")

# Small Model
model_small <- keras_model_sequential()
model_small$add(layer_dense(units = 16, activation = "sigmoid", input_shape = list(1)))
model_small$add(layer_dense(units = 1))
model_small$compile(loss = "mse", optimizer = optimizer_rmsprop(), metrics = list("mae"))

model_small$fit(
  x = x_matrix,
  y = y_matrix,
  epochs = as.integer(100),
  batch_size = as.integer(64),
  verbose = 0
)

# Large Model
model_large <- keras_model_sequential()
model_large$add(layer_dense(units = 128, activation = "sigmoid", input_shape = list(1)))
model_large$add(layer_dense(units = 1))
model_large$compile(loss = "mse", optimizer = optimizer_rmsprop(), metrics = list("mae"))

model_large$fit(
  x = x_high_matrix,
  y = y_high_matrix,
  epochs = as.integer(150),
  batch_size = as.integer(64),
  verbose = 0
)

# Plot approximation
pred_small <- model_small$predict(x_matrix)
pred_large <- model_large$predict(x_matrix)

library(ggplot2)
df_plot <- tibble(x = x, sinx = y, pred_small = as.numeric(pred_small), pred_large = as.numeric(pred_large))

ggplot(df_plot, aes(x = x)) +
  geom_line(aes(y = sinx, color = "True sin(x)"), linewidth = 1.2, linetype = "dashed") +
  geom_line(aes(y = pred_small, color = "Small NN (16 units)"), linewidth = 1) +
  geom_line(aes(y = pred_large, color = "Large NN (128 units)"), linewidth = 1) +
  labs(title = "Sin(x) - NN Comparison", y = "y", color = "Legend") +
  theme_minimal() +
  scale_color_manual(values = c("True sin(x)" = "black", "Small NN (16 units)" = "blue", "Large NN (128 units)" = "red"))


#Portfolio Simulation
simulate_portfolio_returns <- function(data, actor_model) {
  dates <- sort(unique(data$date))
  portfolio_returns <- c()
  
  for (t in 2:length(dates)) {
    df_prev <- data %>% filter(date == dates[t - 1])
    df_curr <- data %>% filter(date == dates[t])
    
    if (nrow(df_prev) < 1 || nrow(df_curr) < 1) next
    
    state <- get_state_tensor(df_prev)
    weights <- as.numeric(actor_model(state))
    weights <- weights / sum(abs(weights))  # normalize
    
    returns <- df_curr$R1M_Usd
    if (length(weights) == length(returns)) {
      portfolio_returns <- c(portfolio_returns, sum(weights * returns, na.rm = TRUE))
    }
  }
  
  return(portfolio_returns)
}

# set metrics and formulas
library(PerformanceAnalytics)
calculate_metrics <- function(portfolio_returns) {
    ret_xts <- xts::xts(portfolio_returns, order.by = seq.Date(from = as.Date("2020-01-02"), by = "month", length.out = length(portfolio_returns)))
    sharpe <- SharpeRatio.annualized(ret_xts)
    mdd <- maxDrawdown(ret_xts)
    ann_vol <- sd(ret_xts) * sqrt(12)
    calmar <- sharpe / mdd
    cum_return <- prod(1 + portfolio_returns) - 1
    
    tibble(
      `Cumulative Return` = cum_return,
      `Annualized Volatility` = ann_vol,
      `Sharpe Ratio` = sharpe,
      `Max Drawdown` = mdd,
      `Calmar Ratio` = calmar
    )
  }

portfolio_returns <- simulate_portfolio_returns(drl_data, trained_drl$actor)
metrics_table <- calculate_metrics(portfolio_returns)
print(metrics_table)


# resets session for libraries
k_clear_session <- function() {
  tryCatch({ keras::backend()$clear_session() }, error = function(e) NULL)
}

# Build Actors, critics, and perform soft update to update weights
build_td3_actor <- function(input_dim) {
  input <- layer_input(shape = input_dim)
  output <- input %>%
    layer_dense(64, activation = "relu") %>%
    layer_dense(64, activation = "relu") %>%
    layer_dense(1, activation = "tanh")
  keras_model(inputs = input, outputs = output)
}

build_td3_critic <- function(input_dim) {
  input <- layer_input(shape = input_dim)
  output <- input %>%
    layer_dense(64, activation = "relu") %>%
    layer_dense(64, activation = "relu") %>%
    layer_dense(1)
  keras_model(inputs = input, outputs = output)
}

soft_update <- function(target, source, tau = 0.005) {
  tw <- target$get_weights()
  sw <- source$get_weights()
  updated <- Map(function(tw, sw) tau * sw + (1 - tau) * tw, tw, sw)
  target$set_weights(updated)
}

# Tensors from data and reward
get_state_tensor <- function(df) {
  x <- df %>% select(all_of(features_short)) %>% as.matrix()
  tf$convert_to_tensor(x, dtype = tf$float32)
}

get_reward_tensor <- function(df) {
  y <- df$R1M_Usd
  if (is.null(y)) stop("Column 'R1M_Usd' not found.")
  tf$convert_to_tensor(matrix(y, ncol = 1), dtype = tf$float32)
}

# Train
train_td3 <- function(data, input_dim, dates, epochs = 5, policy_delay = 2) {
  k_clear_session()
# Initialize actor and critic networks
  actor <- build_td3_actor(input_dim)
  critic1 <- build_td3_critic(input_dim)
  critic2 <- build_td3_critic(input_dim)
  target_actor <- build_td3_actor(input_dim)
  target_critic1 <- build_td3_critic(input_dim)
  target_critic2 <- build_td3_critic(input_dim)
# Build models by passing dummy input to initialize weights
  dummy_input <- tf$zeros(shape(1L, input_dim))
  actor(dummy_input); critic1(dummy_input); critic2(dummy_input)
  target_actor(dummy_input); target_critic1(dummy_input); target_critic2(dummy_input)
# Copy weights from original networks to target networks
  target_actor$set_weights(actor$get_weights())
  target_critic1$set_weights(critic1$get_weights())
  target_critic2$set_weights(critic2$get_weights())
# Optimize for actor and critic
  actor_opt <- optimizer_adam(0.0005)
  critic1_opt <- optimizer_adam(0.001)
  critic2_opt <- optimizer_adam(0.001)
# Loop through each epoch
  for (epoch in 1:epochs) {
    for (t in 2:length(dates)) {
      df_prev <- data %>% filter(date == dates[t - 1])
      df_curr <- data %>% filter(date == dates[t])
      if (nrow(df_prev) == 0 || nrow(df_curr) == 0) next
# convert to tensor
      state <- get_state_tensor(df_prev)
      next_state <- get_state_tensor(df_curr)
      reward <- get_reward_tensor(df_curr)
      
      # Add random noise
      noise <- tf$random$normal(shape = tf$shape(next_state), stddev = 0.1)
      noisy_action <- tf$clip_by_value(target_actor(next_state, training = TRUE) + noise, -1.0, 1.0)
      # Estimate future Q-values
      target_q1 <- target_critic1(next_state, training = TRUE)
      target_q2 <- target_critic2(next_state, training = TRUE)
      target_q <- tf$minimum(target_q1, target_q2)
      target_value <- reward + 0.99 * target_q
      
      # calculate loss between predicted Q-values and target values
      with(tf$GradientTape(persistent = TRUE) %as% tape, {
        q1 <- critic1(state, training = TRUE)
        q2 <- critic2(state, training = TRUE)
        loss1 <- tf$reduce_mean((q1 - target_value)^2)
        loss2 <- tf$reduce_mean((q2 - target_value)^2)
      })
      
      grads1 <- tape$gradient(loss1, critic1$trainable_variables)
      grads2 <- tape$gradient(loss2, critic2$trainable_variables)
      
      if (!all(sapply(grads1, is.null))) {
        critic1_opt$apply_gradients(purrr::transpose(list(grads1, critic1$trainable_variables)))
      }
      
      if (!all(sapply(grads2, is.null))) {
        critic2_opt$apply_gradients(purrr::transpose(list(grads2, critic2$trainable_variables)))
      }
      
      # actor update
      if (t %% policy_delay == 0) {
        with(tf$GradientTape() %as% tape, {
          action_pred <- actor(state, training = TRUE)
          actor_loss <- -tf$reduce_mean(critic1(state, training = TRUE))
        })
        
        actor_grads <- tape$gradient(actor_loss, actor$trainable_variables)
        if (!all(sapply(actor_grads, is.null))) {
          actor_opt$apply_gradients(purrr::transpose(list(actor_grads, actor$trainable_variables)))
        }
        # soft update target networks
        soft_update(target_actor, actor)
        soft_update(target_critic1, critic1)
        soft_update(target_critic2, critic2)
      }
    }
  }
  
  return(list(actor = actor)) # return the trained actor
}

# run trained data
input_dim <- length(features_short)
dates <- sort(unique(drl_data$date))
trained_td3 <- train_td3(drl_data, input_dim, dates, epochs = 5)

# metrics output
td3_returns <- simulate_portfolio_returns(drl_data, trained_td3$actor)
td3_metrics <- calculate_metrics(td3_returns)
print(td3_metrics)
