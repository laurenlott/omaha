library(shiny)
library(xgboost)
library(dplyr)

# Load the dataset and handle missing values
pbp <- read.csv("final_pbp_data.csv", header = TRUE)

# Taking out the rows that a play did not occur in
#pbp <- pbp[pbp$no_play != 1, ]

#pbp <- na.omit(pbp[, c("quarter", "down", "distance", "yards_to_go", "play_type", "regular_play_type", "seconds_left_in_quarter")])

# Filter out the rows with play_type not equal to 'S'
#pbp <- pbp[pbp$play_type != 'S', ]

#pbp <- pbp[, c("quarter", "down", "distance", "yards_to_go", "seconds_left_in_quarter", "regular_play_type")]

# Define X and y
X <- pbp[, c("quarter", "down", "distance", "yards_to_go", "seconds_left_in_quarter")]
y <- pbp$regular_play_type

# Convert y to a factor with fixed levels
y <- factor(y, levels = unique(y))

# XGBoost
set.seed(42)

# Convert labels to numeric
labels <- as.numeric(y)

# Implement train_test_split
train_indices <- sample(nrow(pbp), nrow(pbp) * 0.7)
X_train <- X[train_indices, ]
X_test <- X[-train_indices, ]
y_train <- labels[train_indices]
y_test <- labels[-train_indices]

# Creating an XGBoost classifier
model <- xgboost(data = as.matrix(X_train), label = y_train, nrounds = 10, objective = "multi:softmax", num_class = length(unique(labels)))

# Save the XGBoost model to a file
saveRDS(model, "model.rds")

# Load the XGBoost model
loaded_model <- readRDS("model.rds")

# UI
ui <- fluidPage(
  tags$head(
    tags$script(
      "$(document).ready(function(){
        // Add placeholder text to numeric input fields
        $('#quarter_input').attr('placeholder', 'Enter the Quarter');
        $('#down_input').attr('placeholder', 'Enter the Down');
        $('#distance_input').attr('placeholder', 'Enter the Distance');
        $('#yards_to_go_input').attr('placeholder', 'Enter the Yards to Go');
        $('#seconds_left_input').attr('placeholder', 'Enter the Seconds Left');
      });"
    )
  ),
  titlePanel("Play Prediction App"),
  numericInput("quarter_input", "Quarter:", value = NULL, min = 1, max = 4, step = 1),
  numericInput("down_input", "Down:", value = NULL, min = 1, max = 4, step = 1),
  numericInput("distance_input", "Distance:", value = NULL),
  numericInput("yards_to_go_input", "Yards to Go:", value = NULL),
  numericInput("seconds_left_input", "Seconds Left in Quarter:", value = NULL),
  actionButton("predict_button", "Predict"),
  verbatimTextOutput("output")
)

server <- function(input, output) {
  # Helper function to make predictions
  get_user_input <- function(quarter, down, distance, yards_to_go, seconds_left_in_quarter) {
    # Validate Quarter and Down values (should be between 1 and 4)
    quarter <- ifelse(quarter < 1, 1, ifelse(quarter > 4, 4, quarter))
    down <- ifelse(down < 1, 1, ifelse(down > 4, 4, down))
    
    # Making a prediction using the loaded model
    user_input <- data.frame(
      quarter = quarter,
      down = down,
      distance = distance,
      yards_to_go = yards_to_go,
      seconds_left_in_quarter = seconds_left_in_quarter
    )
    prediction_encoded <- predict(loaded_model, as.matrix(user_input))
    prediction <- levels(y)[prediction_encoded]
    
    # Generate play message
    play_message <- ifelse(prediction == "P", "You should pass the ball.", "You should run the ball.")
    
    # Format the output
    output_text <- paste("Model Prediction:", prediction, "\n", play_message)
    
    return(output_text)
  }
  
  # Observe click events on the "Predict" button
  observeEvent(input$predict_button, {
    # Render the prediction result
    output$output <- renderText({
      HTML(get_user_input(
        input$quarter_input,
        input$down_input,
        input$distance_input,
        input$yards_to_go_input,
        input$seconds_left_input
      ))
    })
  })
}

# Run the Shiny app
shinyApp(ui, server)




