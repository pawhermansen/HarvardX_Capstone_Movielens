# MovieLens_Script.R
#
# HarvardX DataScience Course,
# Capstone Project part 1: MovieLens Prediction.
#
# R Script to build models and create predictions for the movie ratings test set.
#
# Paw Hermansen
#

# Tictoc package times the runtime of code blocks.
if(!require(tictoc)) install.packages("tictoc", dependencies = TRUE, repos = "http://cran.us.r-project.org")
tic("Total script run time")
tic("Read and prepare the data")

#############################################################
#
# Read the data
#
# This part is pre-written by the course staff and copied verbatim here.
#
#############################################################

#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#############################################################
# This ends the part that is pre-written by the course staff.
#############################################################

toc()

dir.create("data", showWarnings = FALSE)
write.csv(edx, row.names = FALSE, 'data/edx.csv')
write.csv(validation, row.names = FALSE, 'data/validation.csv')


#############################################################
#
# Functions to compute the error measure RMSE and the accuracy
#
#############################################################
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

roundRatings <- function(ratings, timestamps) {
  isIntOnly <- validation$timestamp < 1052944896
  mult <- ifelse(isIntOnly, 1, 2)  # 1: only integer ratings, 2: also half-integer ratings
  minValue <- ifelse(isIntOnly, 1.0, 0.5) # 1: only integer ratings, 0.5: also half-integer ratings
  rounded_ratings <- round(mult * ratings, 0) / mult
  rounded_ratings <- ifelse(rounded_ratings < minValue, minValue, rounded_ratings)
  rounded_ratings <- ifelse(5.0 < rounded_ratings, 5.0, rounded_ratings)
  return(rounded_ratings)
}

accuracy <- function(true_ratings, predicted_ratings) {
  mean(true_ratings == predicted_ratings)
}


#############################################################
#
# Model 1: Y_u,i = mu + epsilon_u,i
#
#############################################################

# Find mu: the overall mean of the ratings in the training data
tic("Model 1, building")
mu <- mean(edx$rating)
toc()

tic("Model 1, validating")
# Predict ratings in the validation dataset
predicted_ratings1 <- validation %>% 
  mutate(pred = mu) %>%
  .$pred

# Round the predicted ratings to integer and half-integer ratings
predicted_rounded_ratings1 <- roundRatings(predicted_ratings1, validation$timestamp)

print("Model 1: Y_u,i = mu")
cat("  RMSE = ", RMSE(validation$rating, predicted_ratings1), "\n")
cat("  Accuracy = ", accuracy(validation$rating, predicted_rounded_ratings1), "\n")

toc()

#############################################################
#
# Model 2: Y_u,i = mu + b_i + epsilon_u,i
#
#############################################################

# Find b_i for each movie: the mean of the part of the rating that is not
# explained by the overall mean mu
tic("Model 2, building")
moviePopularity <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))
toc()

tic("Model 2, validating")
# Predict ratings in the validation dataset
predicted_ratings2 <- validation %>% 
  left_join(moviePopularity, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

# Round the predicted ratings to integer and half-integer ratings
predicted_rounded_ratings2 <- roundRatings(predicted_ratings2, validation$timestamp)

print("Model 2: Y_u,i = mu + b_i + epsilon_i")
cat("  RMSE = ", RMSE(validation$rating, predicted_ratings2), "\n")
cat("  Accuracy = ", accuracy(validation$rating, predicted_rounded_ratings2), "\n")

toc()

#############################################################
#
# Model 3: Y_u,i = mu + b_i + b_u + epsilon_u,i
#
#############################################################

# Find b_u for each user: the mean of the part of the rating that is not
# explained by the overall mean mu and the movie popularity b_i
tic("Model 3, building")
userMildness <- edx %>% 
  left_join(moviePopularity, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
toc()

tic("Model 3, validating")
# Predict the ratings in the validation data
predicted_ratings3 <- validation %>% 
  left_join(moviePopularity, by='movieId') %>%
  left_join(userMildness, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

# Round the predicted ratings to integer and half-integer ratings
predicted_rounded_ratings3 <- roundRatings(predicted_ratings3, validation$timestamp)

print("Model 3: Y_u,i = mu + b_i + b_u + epsilon_i")
cat("  RMSE = ", RMSE(validation$rating, predicted_ratings3), "\n")
cat("  Accuracy = ", accuracy(validation$rating, predicted_rounded_ratings3), "\n")

toc()

#############################################################
#
# Model 4: Y_u,i = mu + b_i + b_u + b_u,g + epsilon_u,i
#
#############################################################

# First split each movie into one row for each genre. then find b_ug for each
# (user, genre): the mean of the part of the rating that is not explained by
# the overall mean mu, the movie popularity b_i, and the user mildness b_u.
tic("Model 4, building")
genrePopularity <- edx %>% 
  separate_rows(genres, sep = "\\|") %>%
  left_join(moviePopularity, by='movieId') %>%
  left_join(userMildness, by='userId') %>%
  group_by(userId, genres) %>%
  summarize(b_ug = mean(rating - mu - b_i - b_u))

# Find a list of all genres
genresList <- edx %>%
  separate_rows(genres, sep = "\\|") %>%
  distinct(genres) %>%
  .$genres

# Add rows with b_ug = 0.0 for all non-rated genres for all users. The completness
# of having all rows makes the validation easier.
# First use the R crossing method to generate all user, genre combinations and set
# b_ug = 0.0 for all rows. Then remove all user, genre combinations that exist in
# genrePopularity and add them again from genrePopularity including the earlier
# computed b_ug's.
genrePopularity <- crossing(userId = edx$userId, genres = genresList) %>%
  mutate(b_ug = 0.0) %>%
  anti_join(genrePopularity, by=c('userId', 'genres')) %>%
  bind_rows(genrePopularity)
toc()

tic("Model 4, validating")
# Predict the ratings in the validation data
predicted_ratings4 <- validation %>%
  separate_rows(genres, sep = "\\|") %>%
  left_join(genrePopularity, by=c("userId", "genres")) %>%
  group_by(userId, movieId) %>%
  summarize(b_ug = mean(b_ug)) %>%
  left_join(moviePopularity, by='movieId') %>%
  left_join(userMildness, by='userId') %>%
  mutate(pred = mu + b_i + b_u + b_ug) %>%
  .$pred

# Round the predicted ratings to integer and half-integer ratings
predicted_rounded_ratings4 <- roundRatings(predicted_ratings4, validation$timestamp)

print("Model 4: Y_u,i = mu + b_i + b_u + b_g + epsilon_i")
cat("  RMSE = ", RMSE(validation$rating, predicted_ratings4), "\n")
cat("  Accuracy = ", accuracy(validation$rating, predicted_rounded_ratings4), "\n")
toc()

toc()
