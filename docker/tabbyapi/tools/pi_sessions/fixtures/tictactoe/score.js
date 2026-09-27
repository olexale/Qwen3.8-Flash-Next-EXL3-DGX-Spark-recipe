// Running score across games, kept in localStorage when it exists.

const KEY = "ttt-score";

export function loadScore(storage = globalThis.localStorage) {
  const empty = { X: 0, O: 0, draws: 0 };
  if (!storage) return empty;
  try {
    return { ...empty, ...JSON.parse(storage.getItem(KEY)) };
  } catch {
    return empty;
  }
}

export function recordResult(score, game) {
  const next = { ...score };
  if (game.winner) {
    next[game.winner] += 1;
  } else if (game.draw) {
    next.O += 1;
  }
  return next;
}

export function saveScore(score, storage = globalThis.localStorage) {
  if (storage) storage.setItem(KEY, JSON.stringify(score));
}
