let cells = new Array(42).fill(0);
// let cells = [1, 1, -1, 0, 1, -1, 1, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0]

// Whos turn to start?
let humanTurn = Math.random() >= 0.5;
showTurn(humanTurn);

if (humanTurn) {
    showStones(cells);
}
else {
    aiPlays();
}

function showStones(cells) {
    cellElements = document.getElementsByClassName("cell")

    for (i = 0; i < cells.length; i++) {
        // Current cell
        cell = cellElements[i];

        // Create a new stone for the current cell
        stone = document.createElement('div');

        // Stone type
        stoneType = cells[i]

        cell.innerHTML = '';

        if (stoneType === 1) {
            stone.className = "stone"
            cell.appendChild(stone);
        }
        else if (stoneType === -1) {
            stone.className = "stone red"
            cell.appendChild(stone);
        }
    }
}

function showWinScreen(message, timeout=1500) {
    let winScreen = document.getElementById('win-screen')

    // Set the message
    winScreen.querySelector("H1").innerHTML = message;
    
    // Show the win screen
    winScreen.classList.add('visible', 'opacity')

    // Remove it after a timeout
    setTimeout(() => winScreen.classList.remove('opacity'), timeout)
    setTimeout(() => winScreen.classList.remove('visible'), timeout + 1000)

    // Clear the stones
    setTimeout(() => {
        cells = new Array(42).fill(0);
        showStones(cells);
    }, timeout + 500)

    // Show confetti
    if (message == 'You won!') {
        confetti.start()
        setTimeout(() => confetti.stop(), timeout)
    }
}

async function aiPlays() {
    // No longer the humans turn
    humanTurn = false;
    showTurn(humanTurn);

    let response = await post('play/ai', {cells: cells});

    // If the response was valid
    if (response.valid) {
        cells = response.cells
        showStones(cells);

        if (response.won) {
            showWinScreen('You lost!', 3500)
        }
        else if (response.draw) {
            showWinScreen('Draw!')
        }
    }

    humanTurn = true;
    showTurn(humanTurn);
}

async function humanPlays(position) {
    if (humanTurn) {
        let response = await post('play/human', {cells: cells, position: parseInt(position) - 1});

        // If the response was valid
        if (response.valid) {
            cells = response.cells
            showStones(cells);

            if (response.won) {
                showWinScreen('You won!')
            }
            else if (response.draw) {
                showWinScreen('Draw!')
            }
            else {
                aiPlays();
            }
        }
    }
}

function showTurn(humanTurn) {
    turns = document.getElementsByClassName("turn")

    if (humanTurn) {
        turns[0].classList.add("disabled")
        turns[1].classList.remove("disabled")
    }
    else {
        turns[0].classList.remove("disabled")
        turns[1].classList.add("disabled")
    }
}

function post(url, data) {
    return new Promise(function (resolve, reject) {
        let xhr = new XMLHttpRequest();
        xhr.open('POST', url);
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.onload = function () {
            if (this.status >= 200 && this.status < 300) {
                resolve(JSON.parse(xhr.response));
            } else {
                reject({
                    status: this.status,
                    statusText: xhr.statusText
                });
            }
        };
        xhr.onerror = function () {
            reject({
                status: this.status,
                statusText: xhr.statusText
            });
        };
        xhr.send(JSON.stringify(data));
    });
}