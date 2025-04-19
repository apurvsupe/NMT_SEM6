import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.ImageView;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.image.Image;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import java.io.*;
import java.net.Socket;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import org.bson.Document;
import java.util.ArrayList;
import com.mongodb.client.model.Updates;
import org.bson.types.ObjectId;

public class Main extends Application {

    private static final String SERVER_HOST = "127.0.0.1"; // Server IP
    private static final int SERVER_PORT = 5000; // Must match the Python server
    private  MongoClient mongoClient;
    private MongoDatabase database;
    private MongoCollection<Document> collection;
    private ObjectId currentConversationId;

   // MODIFIED CODE
    @Override
    public void start(Stage primaryStage) {

        try{
            mongoClient = MongoClients.create("mongodb://localhost:27017/");
            database = mongoClient.getDatabase("NMT");
            collection = database.getCollection("summarization");
            System.out.println("MongoDB connection established");
        }
        catch(Exception e){
            System.err.println("MongoDB connection failed: " + e.getMessage());
            e.printStackTrace();
        }
        // Labels
        Label promptLabel = new Label("Enter Text:");
        Label resultLabel = new Label("Translation:");

        // TextArea for input and output
        TextArea inputField = new TextArea();
        inputField.setPromptText("Enter text here...");
        inputField.setWrapText(true);

        TextArea outputField = new TextArea();
        outputField.setEditable(false);
        outputField.setWrapText(true);

        // Dropdown for language selection
        ComboBox<String> languageSelection = new ComboBox<>();
        languageSelection.getItems().addAll("Marathi to English", "English to Marathi");
        languageSelection.setValue("Marathi to English"); // Default

        // Translate Button
        Button translateButton = new Button("Translate");
        Button newConversation = new Button("New Conversation");
        Button endConversation = new Button("End Conversation");

        // Progress Indicator
        ProgressIndicator progressIndicator = new ProgressIndicator();
        progressIndicator.setVisible(false);

        translateButton.setOnAction(e -> {
            String text = inputField.getText().trim();
            String translationType = languageSelection.getValue();

            if (!text.isEmpty()) {
                progressIndicator.setVisible(true);
                translateButton.setDisable(true);

                // Checking conversation ID
                if(currentConversationId == null){
                    createNewConversation();
                }

                String prefix = translationType.equals("Marathi to English") ? "MR-EN|" : "EN-MR|";
                String translatedText = requestTranslation(prefix + text);

                addTranslationConversation(text, translatedText, translationType);
                outputField.setText(translatedText);
                progressIndicator.setVisible(false);
                translateButton.setDisable(false);
            } else {
                outputField.setText("Please enter some text.");
            }
        });


        newConversation.setOnAction(e->{
            createNewConversation();
            inputField.clear();
            outputField.clear();
            outputField.setText("New conversation started");
        });

        endConversation.setOnAction(e ->{
            if(currentConversationId != null){
                try{
                    // Updating conversation status

                    collection.updateOne(
                            new Document("_id", currentConversationId),
                            new Document("$set", new Document("status", "completed"))
                    );

                    Document conversationDoc = collection.find(new Document("_id", currentConversationId)).first();
                    if(conversationDoc != null) {
                        ArrayList<Document> translations = (ArrayList<Document>) conversationDoc.get("translations");

                        // Joining all the texts in a string variable
                        StringBuilder allTexts = new StringBuilder();
                        if (translations != null) {
                            for (Document translation : translations) {
                                String englishText = translation.getString("text");
                                if (englishText != null && !englishText.isEmpty()) {
                                    allTexts.append(englishText).append("\n");
                                }
                            }
                        }

                        // Sending the joined texts to the model server
                        String joinedTexts = allTexts.toString().trim();
                        if (!joinedTexts.isEmpty()) {
                            String prefix = "END-CON|";
                            joinedTexts = prefix + joinedTexts;
                            System.out.println("After adding prefix: " + joinedTexts);

                            String response = requestConversationSummary(joinedTexts);
                            collection.updateOne(
                                    new Document("_id", currentConversationId),
                                    new Document("$set", new Document("summary", response)));

                            outputField.setText("Conversation ended: " + response);

                        } else {
                            outputField.setText("Conversation ended. No text to process.");
                        }
                    }
                    currentConversationId = null;
                }
                catch(Exception ex){
                    System.err.println("Error ending conversation: " + ex.getMessage());
                    ex.printStackTrace();
                }
            }
        });


        // Left Alignemnt for Text Field and TextBoxArea
        VBox textBox = new VBox(5, promptLabel, inputField, resultLabel, outputField);
        textBox.setAlignment(Pos.CENTER_LEFT);
        textBox.setPadding(new Insets(10));

        // HBox Layout for right-alignment of buttons
        VBox buttonBox = new VBox(5, languageSelection, translateButton, newConversation, endConversation);
        buttonBox.setAlignment(Pos.CENTER_RIGHT);
        buttonBox.setPrefWidth(200);

        languageSelection.setPrefWidth(150);
        translateButton.setPrefWidth(150);
        newConversation.setPrefWidth(150);
        endConversation.setPrefWidth(150);


        // Logo Image
        Image logoImage = new Image(getClass().getResourceAsStream("/logo.jpg"));
        ImageView logo = new ImageView(logoImage);
        logo.setFitWidth(200);
        logo.setPreserveRatio(true);

        // Logo Alignment
        HBox topBar = new HBox(10, logo);
        topBar.setAlignment(Pos.TOP_LEFT);
        topBar.setPadding(new Insets(10));



        GridPane layout = new GridPane();
        layout.setPadding(new Insets(10));
        layout.setVgap(7);
        layout.setHgap(20);
        layout.add(textBox, 0, 0);
        layout.add(buttonBox, 1, 0);

//        Scene scene = new Scene(layout, 500, 350);
        Scene scene = new Scene(layout, 600, 400);
        scene.getStylesheets().add(getClass().getResource("style.css").toExternalForm()); 
        primaryStage.setScene(scene);
        primaryStage.setScene(scene);
        primaryStage.setTitle("Language Translator");
        primaryStage.setResizable(false);
        primaryStage.getIcons().add(logoImage);
        primaryStage.show();
    }



    @Override
    public void stop(){
        if(mongoClient != null){
            mongoClient.close();
            System.out.println("MongoDB connection closed");
        }
    }


    private void addTranslationConversation(String originalText, String translatedText, String translationTyep){
        try{
            String englishText;
            if(translationTyep.equals("English to Marathi")){
                englishText = originalText;
            }
            else{
                englishText = translatedText;
            }

            Document translationDoc = new Document("timestamp", new java.util.Date())
                    .append("text", englishText);

            collection.updateOne(
                    new Document("_id", currentConversationId),
                    new Document("$push", new Document("translations", translationDoc))
            );

        }
        catch(Exception e){
            System.err.println("Error adding translation to conversation: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void createNewConversation(){
        try{
            Document conversationDoc = new Document("timestamp", new java.util.Date())
                    .append("status", "active")
                    .append("translations", new ArrayList<Document>());

            collection.insertOne(conversationDoc);
            currentConversationId = conversationDoc.getObjectId("_id");
            System.out.println("New conversation created with ID: " + currentConversationId);

        }

        catch(Exception e){
            System.err.println("Error creating new conversation: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private String requestConversationSummary(String joinedTexts){
        String response = "No response";
        System.out.println("Reached Here");

        try(Socket socket = new Socket(SERVER_HOST, SERVER_PORT);
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))
        ){

            out.println(joinedTexts);

            response = in.readLine();
            System.out.println("End conversation response: "+ response);
        }
        catch(Exception ex){
            ex.printStackTrace();
            response = "Error connecting to the server.";
        }

        return response;
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
            System.out.println("Response from Python server: " + response + " " + text);



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
