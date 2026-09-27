# Scripted pi sessions for real-traffic A/B checks (run by ../pi_sessions.py)

Each `## task <name>` is one pi session; its turns are separated by lines that hold only `---`
and go to the same session one after another. `fixture: <dir>` copies `fixtures/<dir>` into the
session's empty working folder first; without it the folder starts empty.

The tasks mix what a normal session does: reading and reviewing code (thinking, prose), writing
new files, and editing files already in the context (the edit tool calls where prompt lookup can
help). They are sized to finish in a few minutes each, and they only need commands the owner's
pi permission config allows without asking (`node`, `ls`, `cat`, `mkdir`, ...): no git, no
servers, no package installs. Change a prompt and the results are no longer comparable with
earlier runs, so add a new task instead of editing one that has results.

## task review
fixture: tictactoe

Review this small tic-tac-toe web app (index.html, ui.js, game.js, score.js, game.test.js).
List the bugs you find, with file and line, most serious first. Don't change any files yet.
Don't use git, don't start a server and don't open a browser.

---

Fix all of them. For each bug in game.js or score.js, add a test (in game.test.js, or a new
score.test.js) that fails without the fix. Run the tests with `node --test` and make sure they
pass.

## task snake

In this empty folder, build a small Snake game for the browser: plain JavaScript ES modules, no
dependencies, no build step. Put the rules in snake.js (a 16x16 grid, pure functions, no DOM),
the rendering and keyboard handling in ui.js (a canvas, arrow keys), plus index.html and a
package.json with "type": "module". Keep it short: snake.js under about 100 lines. Don't use
git, don't start a server and don't open a browser.

---

Add a score (one point per food) and a high score kept in localStorage, both shown above the
board.

---

Add pause and resume on P or Space, and make the snake 10% faster after every 5 food eaten.

---

Write snake.test.js with node:test covering movement, growing after eating, food never placed
on the snake, and game over on hitting a wall or itself. Run `node --test` and fix whatever
fails.
