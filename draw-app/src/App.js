import logo from "./logo.svg";
import Draw from "./components/Draw";
import "./App.css";

function App() {
  return (
    <div className="App">
      <h3 className="text">Classify Handwritten Digits</h3>
      <p></p>
      <div>
        <Draw />
      </div>
    </div>
  );
}

export default App;
