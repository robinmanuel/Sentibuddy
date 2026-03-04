import { useState, useRef, useEffect } from "react";
import "./App.css";

// Get the current host IP address for backend connection
const getBackendUrl = () => {
  const hostname = window.location.hostname;
  // If running on localhost, use localhost; otherwise use the network IP
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return "http://localhost:5000";
  }
  return `http://${hostname}:5000`;
};

const API_URL = getBackendUrl();
console.log("Backend API URL:", API_URL);

const apiService = {
  analyzeText: async (text) => {
    const response = await fetch(`${API_URL}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  },

  checkHealth: async () => {
    try {
      console.log("Checking health at:", `${API_URL}/health`);
      const response = await fetch(`${API_URL}/health`);
      console.log("Health check response:", response);
      const data = await response.json();
      console.log("Health check data:", data);
      return data;
    } catch (error) {
      console.error("Health check failed:", error);
      return { status: "error", apis: { gemini: "unavailable" } };
    }
  },
};

function App() {
  const [userInput, setUserInput] = useState("");
  const [messages, setMessages] = useState([
    {
      type: "bot",
      content:
        "Hi there! How are you feeling today? Share your thoughts with me.",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState({ status: "unknown" });
  const messagesEndRef = useRef(null);

  useEffect(() => {
    const checkApiHealth = async () => {
      const status = await apiService.checkHealth();
      setApiStatus(status);

      if (status.status !== "ok") {
        setMessages((prev) => [
          ...prev,
          {
            type: "bot",
            content:
              "I'm having trouble connecting to my backend services. Some features might be limited.",
            isError: true,
          },
        ]);
      }
    };

    checkApiHealth();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const analyzeSentiment = async (e) => {
    e.preventDefault();

    if (!userInput.trim()) return;

    setMessages((prev) => [...prev, { type: "user", content: userInput }]);

    const currentInput = userInput;
    setUserInput("");

    setIsLoading(true);

    try {
      await new Promise((resolve) => setTimeout(resolve, 500));

      const data = await apiService.analyzeText(currentInput);

      setMessages((prev) => [
        ...prev,
        {
          type: "bot",
          content: data.recommendation,
          sentiment: data.sentiment,
          emoji: getEmoji(data.sentiment),
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          type: "bot",
          content: "Sorry, I had trouble analyzing that. Can you try again?",
          isError: true,
        },
      ]);
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const getEmoji = (sentiment) => {
    if (!sentiment) return "";

    const emojis = {
      happy: "😊",
      joyful: "😄",
      excited: "🤩",
      content: "🙂",
      surprised: "😮",
      curious: "🤔",
      sad: "😢",
      disappointed: "😔",
      angry: "😠",
      fearful: "😨",
      anxious: "😰",
      neutral: "😐",
    };

    return emojis[sentiment.toLowerCase()] || "";
  };

  const getEmotionColor = (sentiment) => {
    if (!sentiment) return "";

    const colors = {
      happy: "#4caf50",
      joyful: "#8bc34a",
      excited: "#ffc107",
      content: "#2196f3",
      surprised: "#9c27b0",
      curious: "#00bcd4",
      sad: "#607d8b",
      disappointed: "#9e9e9e",
      angry: "#f44336",
      fearful: "#ff5722",
      anxious: "#ff9800",
      neutral: "#9e9e9e",
    };

    return colors[sentiment.toLowerCase()] || "";
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <div className="logo-container">SentiBuddy</div>
        {apiStatus.status !== "ok" && (
          <div
            className="api-status-indicator offline"
            title="Backend service unavailable"
          >
            ⚠️
          </div>
        )}
      </div>

      <div className="messages-container">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`message ${message.type} ${
              message.isError ? "error" : ""
            }`}
          >
            {message.type === "bot" && (
              <div className="avatar">
                <span role="img" aria-label="bot">
                  🤖
                </span>
              </div>
            )}
            <div className="bubble">
              <p>{message.content}</p>
              {message.sentiment && (
                <div className={`emotion-tag ${message.sentiment}`}>
                  Detected: {message.sentiment} {message.emoji}
                </div>
              )}
            </div>
            {message.type === "user" && (
              <div className="avatar">
                <span role="img" aria-label="user">
                  👤
                </span>
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="message bot">
            <div className="avatar">
              <span role="img" aria-label="bot">
                🤖
              </span>
            </div>
            <div className="bubble typing">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={analyzeSentiment} className="input-form">
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Tell me how you're feeling..."
          className="chat-input"
          disabled={apiStatus.status !== "ok"}
        />
        <button
          type="submit"
          className="send-button"
          disabled={!userInput.trim() || isLoading || apiStatus.status !== "ok"}
        >
          <span role="img" aria-label="send">
            ➤
          </span>
        </button>
      </form>
    </div>
  );
}

export default App;
