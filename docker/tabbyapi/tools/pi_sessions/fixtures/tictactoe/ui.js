import { newGame, play, undo, status } from "./game.js";
import { loadScore, recordResult, saveScore } from "./score.js";

const boardEl = document.querySelector("#board");
const statusEl = document.querySelector("#status");
const scoreEl = document.querySelector("#score");

let game = newGame();
let score = loadScore();
let starter = "X";

function render() {
  boardEl.innerHTML = "";
  game.board.forEach((mark, i) => {
    const btn = document.createElement("button");
    btn.className = "cell";
    btn.textContent = mark ?? "";
    if (game.line && game.line.includes(i)) btn.classList.add("win");
    btn.addEventListener("click", () => onCell(i));
    boardEl.append(btn);
  });
  statusEl.textContent = status(game);
  scoreEl.textContent = `X ${score.X} · O ${score.O} · draws ${score.draws}`;
}

function onCell(i) {
  const before = game;
  game = play(game, i);
  if (game !== before && (game.winner || game.draw)) {
    score = recordResult(score, game);
    saveScore(score);
  }
  render();
}

document.querySelector("#undo").addEventListener("click", () => {
  game = undo(game);
  render();
});

document.querySelector("#new").addEventListener("click", () => {
  starter = starter === "X" ? "O" : "X";
  game = newGame();
  render();
});

render();
