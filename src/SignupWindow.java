import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import com.mongodb.client.*;
import org.bson.Document;

public class SignupWindow extends Application {

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Sign Up");

        Label nameLabel = new Label("Full Name:");
        TextField nameField = new TextField();

        Label userLabel = new Label("Username:");
        TextField userField = new TextField();

        Label passLabel = new Label("Password:");
        PasswordField passField = new PasswordField();

        Button registerButton = new Button("Register");
        registerButton.getStyleClass().add("custom-button");

        Label messageLabel = new Label();

        registerButton.setOnAction(e -> {
            String fullName = nameField.getText();
            String username = userField.getText();
            String password = passField.getText();

            if (username.isEmpty() || password.isEmpty()) {
                messageLabel.setText("Please fill all fields.");
                return;
            }


            try (MongoClient mongoClient = MongoClients.create("mongodb://localhost:27017")) {
                MongoDatabase database = mongoClient.getDatabase("NMT");
                MongoCollection<Document> collection = database.getCollection("users");

                // Check if user already exists
                Document existingUser = collection.find(new Document("username", username)).first();
                if (existingUser != null) {
                    messageLabel.setText("Username already exists.");
                    return;
                }

                Document doc = new Document("fullname", fullName)
                        .append("username", username)
                        .append("password", password);
                collection.insertOne(doc);

                // Redirect to login window
                messageLabel.setText("Registration Successful!");


                new Thread(() -> {
                    try {
                        Thread.sleep(1000);
                    } catch (InterruptedException ignored) {}
                    javafx.application.Platform.runLater(() -> {
                        // Open login window
                        LoginWindow loginWindow = new LoginWindow();
                        Stage loginStage = new Stage();
                        try {
                            loginWindow.start(loginStage);
                        } catch (Exception ex) {
                            ex.printStackTrace();
                        }
                        primaryStage.close(); // Close signup window
                    });
                }).start();

            } catch (Exception ex) {
                messageLabel.setText("Error: " + ex.getMessage());
            }
        });

        VBox vbox = new VBox(10, nameLabel, nameField, userLabel, userField, passLabel, passField, registerButton, messageLabel);
        vbox.setAlignment(Pos.CENTER);
        vbox.setPadding(new Insets(20));

        Scene scene = new Scene(vbox, 300, 350);
        scene.getStylesheets().add(getClass().getResource("layer.css").toExternalForm());

        primaryStage.setScene(scene);
        primaryStage.setResizable(false);
        primaryStage.show();
    }
}
