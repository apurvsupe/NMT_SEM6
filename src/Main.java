import javafx.application.Application;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import java.io.*;
import java.net.Socket;

public class Main extends Application {

    private static final String SERVER_HOST = "127.0.0.1"; // Server IP
    private static final int SERVER_PORT = 5000; // Must match the Python server

    @Override
    public void start(Stage primaryStage) {
        Label promptLabel = new Label("Enter Text:");
        TextField inputField = new TextField();
        Button translateButton = new Button("Translate");
        Label resultLabel = new Label("Translation: ");

        // Dropdown for language selection
        ComboBox<String> languageSelection = new ComboBox<>();
        languageSelection.getItems().addAll("Marathi to English", "English to Marathi");
        languageSelection.setValue("Marathi to English");  // Default

        translateButton.setOnAction(e -> {
            String text = inputField.getText().trim();
            String translationType = languageSelection.getValue();

            if (!text.isEmpty()) {
                // Set direction based on selection
                String prefix = translationType.equals("Marathi to English") ? "MR-EN|" : "EN-MR|";
                String translatedText = requestTranslation(prefix + text);
                resultLabel.setText("Translation: " + translatedText);
            } else {
                resultLabel.setText("Please enter some text.");
            }
        });

        VBox layout = new VBox(10, promptLabel, inputField, languageSelection, translateButton, resultLabel);
        layout.setAlignment(Pos.CENTER);
        Scene scene = new Scene(layout, 400, 250);
        primaryStage.setScene(scene);
        primaryStage.setTitle("Language Translator");
        primaryStage.show();
    }

    private String requestTranslation(String text) {
        String response = "No response";
        try (Socket socket = new Socket(SERVER_HOST, SERVER_PORT);
             PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
             BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {

            // Send text to Python server
            out.println(text);

            // Receive translated text from the server
            response = in.readLine();

        } catch (Exception e) {
            e.printStackTrace();
            response = "Error connecting to the server.";
        }
        return response;
    }

    public static void main(String[] args) {
        launch(args);
    }
}
