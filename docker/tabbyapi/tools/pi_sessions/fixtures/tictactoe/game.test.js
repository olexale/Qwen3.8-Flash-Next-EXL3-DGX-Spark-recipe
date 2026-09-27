import { test } from "node:test";
import assert from "node:assert/strict";
import { newGame, play, undo, status } from "./game.js";

const playAll = (cells, g = newGame()) => cells.reduce(play, g);

test("X moves first and turns alternate", () => {
  const g = playAll([4]);
  assert.equal(g.board[4], "X");
  assert.equal(g.current, "O");
});

test("an occupied cell is ignored", () => {
  const g = playAll([4, 4]);
  assert.equal(g.current, "O");
  assert.deepEqual(g.moves, [4]);
});

test("a row wins", () => {
  const g = playAll([0, 3, 1, 4, 2]);
  assert.equal(g.winner, "X");
  assert.equal(status(g), "X wins");
});

test("undo takes back the last move", () => {
  const g = undo(playAll([0, 1]));
  assert.deepEqual(g.moves, [0]);
  assert.equal(g.current, "O");
});
