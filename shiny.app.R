
#shiny dashboard 

# predicted probabilities 
probs <- predict(rf_smote, newdata = test_copy, type = "prob")[, "Yes"]
true <- test$attrition

ui <- fluidPage(
  titlePanel("Attrition Model Threshold Explorer"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("thresh", "Prediction Threshold", min = 0.1, max = 0.9, value = 0.3, step = 0.01),
      verbatimTextOutput("f1")
    ),
    mainPanel(
      plotOutput("rocPlot"),
      tableOutput("confMatrix")
    )
  )
)

server <- function(input, output) {
  
  output$rocPlot <- renderPlot({
    plot(roc(true, probs), main = "ROC Curve", col = "blue", lwd = 2)
    auc_val <- auc(roc(true, probs))
    text(0.6, 0.4, paste("AUC =", round(auc_val, 3)), col = "blue")
  })
  
  output$confMatrix <- renderTable({
    pred <- factor(ifelse(probs > input$thresh, "Yes", "No"), levels = c("No", "Yes"))
    cm <- caret::confusionMatrix(pred, true, positive = "Yes")
    cm$table
  })
  
  output$f1 <- renderText({
    pred <- factor(ifelse(probs > input$thresh, "Yes", "No"), levels = c("No", "Yes"))
    cm <- caret::confusionMatrix(pred, true, positive = "Yes")
    paste("F1 Score:", round(cm$byClass["F1"], 3))
  })
  
}

# Launch the app
shinyApp(ui = ui, server = server)