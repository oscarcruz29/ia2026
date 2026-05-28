const vscode  = require("vscode");
const path    = require("path");
const { spawn } = require("child_process");
const readline  = require("readline");

let proc    = null;
let rl      = null;
let pending = new Map();
let reqId   = 0;

function ensureServer() {
  if (proc) return;

  const root = "C:\\Users\\oscar\\OneDrive\\Documentos\\GitHub\\ia2026\\proyecto3";
  const pythonExe = root + "\\venv\\Scripts\\python.exe";
  const scriptPath = root + "\\rnn-zak\\server_stdio.py";

  // Ejecutamos Python de forma directa, sin shell: true
  // Pasamos -u para asegurar que no haya buffer
  proc = spawn(pythonExe, ["-u", scriptPath], {
    cwd: root,
    stdio: ["pipe", "pipe", "pipe"],
    env: {
      ...process.env,
      "PYTHONUNBUFFERED": "1",
      "TF_CPP_MIN_LOG_LEVEL": "3", // Silencia los warnings de TensorFlow en consola
      "CUDA_VISIBLE_DEVICES": "-1" // Fuerza uso de CPU
    }
  });

  rl = readline.createInterface({ input: proc.stdout });

  rl.on("line", (line) => {
    try {
      const msg = JSON.parse(line);
      const cb  = pending.get(msg._id);
      if (cb) {
        pending.delete(msg._id);
        cb(msg);
      }
    } catch (_) {}
  });

  proc.stderr.on("data", (d) => {
    console.log("[zak-rnn stderr]", d.toString().trim());
  });

  proc.on("error", (err) => {
    vscode.window.showErrorMessage("Zak RNN error al spawnear: " + err.message);
    proc = null;
  });

  proc.on("exit", (code) => {
    if (code !== 0 && code !== null) {
      vscode.window.showErrorMessage(`Zak RNN: El servidor Python se cerró (Código ${code})`);
    }
    proc = null;
    rl   = null;
    
    for (const [id, cb] of pending.entries()) {
      cb({ ok: false, error: "Servidor cerrado inesperadamente" });
    }
    pending.clear();
  });
}

function request(method, fields) {
  return new Promise((resolve, reject) => {
    ensureServer();
    if (!proc) {
      reject(new Error("Servidor no iniciado"));
      return;
    }

    const id    = ++reqId;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error("Timeout esperando al servidor zak-rnn"));
    }, 120000);

    pending.set(id, (msg) => {
      clearTimeout(timer);
      if (msg.ok) resolve(msg);
      else reject(new Error(msg.error || "error desconocido"));
    });

    // Añadimos \r\n para forzar la lectura de línea en Windows
    proc.stdin.write(JSON.stringify({ method, _id: id, ...fields }) + "\r\n");
  });
}

async function zakComplete() {
  const ed = vscode.window.activeTextEditor;
  if (!ed) return;

  const pos    = ed.selection.active;
  const prefix = ed.document.lineAt(pos.line).text.slice(0, pos.character);
  const cfg    = vscode.workspace.getConfiguration("zakRnn");
  const maxNew = cfg.get("maxNew")      || 60;
  const temp   = cfg.get("temperature") || 0.75;

  try {
    const res    = await request("complete", { prefix, max_new: maxNew, temperature: temp });
    const suffix = res.text.slice(prefix.length);
    if (!suffix) {
      vscode.window.showInformationMessage("Zak RNN: no se generó sufijo nuevo.");
      return;
    }
    await ed.edit((eb) => eb.insert(pos, suffix));
  } catch (err) {
    vscode.window.showErrorMessage("Zak RNN complete: " + err.message);
  }
}

async function zakSuggest() {
  const ed = vscode.window.activeTextEditor;
  if (!ed) return;

  const pos    = ed.selection.active;
  const prefix = ed.document.lineAt(pos.line).text.slice(0, pos.character);

  try {
    const res  = await request("suggest", { prefix, n: 5 });
    if (!res.items || res.items.length === 0) {
      vscode.window.showInformationMessage("Zak RNN: sin sugerencias.");
      return;
    }
    const pick = await vscode.window.showQuickPick(res.items, {
      placeHolder: "Sugerencias Zak RNN — elige una",
    });
    if (!pick) return;
    await ed.edit((eb) => eb.insert(pos, pick.slice(prefix.length)));
  } catch (err) {
    vscode.window.showErrorMessage("Zak RNN suggest: " + err.message);
  }
}

function activate(ctx) {
  ctx.subscriptions.push(
    vscode.commands.registerCommand("zakRnn.complete", zakComplete),
    vscode.commands.registerCommand("zakRnn.suggest",  zakSuggest),
  );
  console.log("[zak-rnn] extension activada");
}

function deactivate() {
  if (proc) proc.kill();
}

module.exports = { activate, deactivate };