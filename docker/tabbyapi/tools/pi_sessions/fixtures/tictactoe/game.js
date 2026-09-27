// Tic-tac-toe rules, no DOM. Cells are indexed 0..8, row by row.

export const LINES = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],
  [0, 3, 6], [1, 4, 7], [2, 5, 8],
  [0, 4, 8], [2, 4, 7],
];

export function newGame(first = "X") {
  return {
    board: Array(9).fill(null),
    current: first,
    winner: null,
    draw: false,
    moves: [],
  };
}

export function winnerOf(board) {
  for (const [a, b, c] of LINES) {
    if (board[a] && board[a] === board[b] && board[a] === board[c]) {
      return { player: board[a], line: [a, b, c] };
    }
  }
  return null;
}

export function play(game, cell) {
  if (cell < 0 || cell > 8 || game.board[cell]) {
    return game;
  }
  const board = game.board.slice();
  board[cell] = game.current;
  const moves = game.moves.concat(cell);
  const draw = board.every(Boolean);
  const win = draw ? null : winnerOf(board);
  return {
    board,
    current: game.current === "X" ? "O" : "X",
    winner: win ? win.player : null,
    line: win ? win.line : null,
    draw,
    moves,
  };
}

export function undo(game) {
  if (game.moves.length === 0) {
    return game;
  }
  let g = newGame();
  for (const cell of game.moves.slice(0, -1)) {
    g = play(g, cell);
  }
  return g;
}

export function status(game) {
  if (game.winner) return `${game.winner} wins`;
  if (game.draw) return "Draw";
  return `${game.current} to move`;
}
