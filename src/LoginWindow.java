import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import com.mongodb.client.*;
import org.bson.Document;

public class LoginWindow extends Application {

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Login");

        Label userLabel = new Label("Username:");
        TextField userField = new TextField();

        Label passLabel = new Label("Password:");
        PasswordField passField = new PasswordField();

        Button loginButton = new Button("Login");
        Button signupButton = new Button("Sign Up");
        Label messageLabel = new Label();

        loginButton.getStyleClass().add("custom-button");
        signupButton.getStyleClass().add("custom-button");

        loginButton.setOnAction(e -> {
            String username = userField.getText();
            String password = passField.getText();

            // Connect to MongoDB and validate user credentials
            try (MongoClient mongoClient = MongoClients.create("mongodb://localhost:27017")) {
                MongoDatabase database = mongoClient.getDatabase("NMT");
                MongoCollection<Document> collection = database.getCollection("users");

                // Find user by username
                Document user = collection.find(new Document("username", username)).first();

                if (user != null && user.getString("password").equals(password)) {
                    messageLabel.setText("Login Successful!");


                    Main mainClass = new Main();
                    Stage mainStage = new Stage();
                    try {
                        mainClass.start(mainStage);
                    } catch (Exception ex) {
                        ex.printStackTrace();
                    }
                    primaryStage.close();
                } else {
                    messageLabel.setText("Invalid Credentials");
                }
            } catch (Exception ex) {
                messageLabel.setText("Error: " + ex.getMessage());
            }
        });

        signupButton.setOnAction(e -> {
            SignupWindow signupWindow = new SignupWindow();
            Stage signupStage = new Stage();
            try {
                signupWindow.start(signupStage);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
            primaryStage.close(); // Close the login window
        });

        VBox vbox = new VBox(10, userLabel, userField, passLabel, passField, loginButton, signupButton, messageLabel);
        vbox.setAlignment(Pos.CENTER);
        vbox.setPadding(new Insets(20));

        Scene scene = new Scene(vbox, 300, 300);
        scene.getStylesheets().add(getClass().getResource("layer.css").toExternalForm());

        primaryStage.setScene(scene);
        primaryStage.show();
        primaryStage.setResizable(false);
    }

    public static void main(String[] args) {
        launch(args);
    }
}
