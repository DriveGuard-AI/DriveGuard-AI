'use strict'

/**
 * DriveGuard AI Electron 主进程。
 *
 * 职责边界：
 * - 读取打包时内置的 `config.json`（模型服务地址/Token），不向用户暴露配置界面；
 * - 生成每次启动随机的本地管理 Token，拉起 PyInstaller 后端；
 * - 等 `/api/v1/health` 就绪后打开无地址栏的主窗口；
 * - 退出时关闭后端，把后端日志写入用户数据目录。
 *
 * 安全基线：渲染进程 sandbox + contextIsolation + 无 nodeIntegration；
 * 后端只监听 127.0.0.1；模型服务 Token 只写入用户目录文件，不进入渲染进程。
 */

const { app, BrowserWindow, Menu, dialog, ipcMain, shell } = require('electron')
const { spawn } = require('node:child_process')
const crypto = require('node:crypto')
const fs = require('node:fs')
const http = require('node:http')
const https = require('node:https')
const net = require('node:net')
const path = require('node:path')

const APP_ID = 'ai.driveguard.desktop'
const isDemo = process.argv.includes('--demo') || process.env.DGB_DESKTOP_DEMO === '1'

// 固定用户数据目录，避免 Electron 默认按 package name 生成不同路径。
app.setPath('userData', path.join(app.getPath('appData'), 'DriveGuardAI'))

let mainWindow = null
let splashWindow = null
let backendProcess = null
let backendLog = null
let backendError = null
let booting = true
let isQuitting = false
let localToken = ''
let backendPort = 0
let backendBase = ''

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function repoRoot() {
  return app.isPackaged ? process.resourcesPath : path.resolve(__dirname, '..', '..')
}

function bundledFrontend() {
  return app.isPackaged
    ? path.join(process.resourcesPath, 'frontend')
    : path.join(repoRoot(), 'frontend', 'dist')
}

function configPath() {
  return app.isPackaged
    ? path.join(process.resourcesPath, 'config.json')
    : path.join(__dirname, '..', 'config.json')
}

function logFile() {
  return path.join(app.getPath('userData'), 'logs', 'backend.log')
}

function appendLog(text) {
  if (backendLog) backendLog.write(text)
}

function rotateLog() {
  const file = logFile()
  try {
    if (fs.existsSync(file) && fs.statSync(file).size > 5 * 1024 * 1024) {
      fs.renameSync(file, `${file}.1`)
    }
  } catch {
    // 日志轮转失败不能阻止启动。
  }
}

function loadConfig() {
  if (isDemo) return { demo: true }
  const file = configPath()
  if (!fs.existsSync(file)) {
    throw new Error(`缺少内置模型服务配置：${file}`)
  }
  let config
  try {
    config = JSON.parse(fs.readFileSync(file, 'utf8'))
  } catch (error) {
    throw new Error(`内置配置不是合法 JSON：${error.message}`)
  }
  if (typeof config.modelServiceUrl !== 'string' || !/^https?:\/\//i.test(config.modelServiceUrl)) {
    throw new Error('内置配置 modelServiceUrl 必须是 http(s) 地址')
  }
  if (typeof config.modelServiceToken !== 'string' || config.modelServiceToken.trim().length < 24) {
    throw new Error('内置配置 modelServiceToken 至少需要 24 个字符')
  }
  return config
}

function findFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.unref()
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      const port = typeof address === 'object' && address ? address.port : 0
      server.close(() => (port ? resolve(port) : reject(new Error('无法分配本地端口'))))
    })
  })
}

function checkHealth(port) {
  return new Promise((resolve, reject) => {
    const request = http.get(
      { host: '127.0.0.1', port, path: '/api/v1/health', timeout: 1500 },
      (response) => {
        response.resume()
        if (response.statusCode === 200) resolve()
        else reject(new Error(`health HTTP ${response.statusCode}`))
      },
    )
    request.on('timeout', () => request.destroy(new Error('health timeout')))
    request.on('error', reject)
  })
}

async function waitForHealth(port, timeoutMs = 45000) {
  const deadline = Date.now() + timeoutMs
  let lastError = null
  while (Date.now() < deadline) {
    if (backendError) throw new Error(`本地后端无法启动：${backendError}`)
    if (!backendProcess) {
      throw new Error('本地后端进程已退出，请查看日志')
    }
    try {
      await checkHealth(port)
      return
    } catch (error) {
      lastError = error
      await delay(300)
    }
  }
  throw new Error(`本地后端启动超时：${lastError ? lastError.message : '未知原因'}`)
}

function probeModelService(config, timeoutMs = 5000) {
  return new Promise((resolve) => {
    let target
    try {
      target = new URL(`${config.modelServiceUrl.replace(/\/+$/, '')}/api/live/health`)
    } catch {
      resolve({ ok: false, reason: '模型服务地址不是合法 URL' })
      return
    }
    const client = target.protocol === 'https:' ? https : http
    const request = client.get(
      target,
      {
        headers: { Authorization: `Bearer ${config.modelServiceToken}` },
        timeout: timeoutMs,
      },
      (response) => {
        response.resume()
        resolve({
          ok: response.statusCode === 200,
          reason: `HTTP ${response.statusCode}`,
        })
      },
    )
    request.on('timeout', () => request.destroy(new Error('连接超时')))
    request.on('error', (error) => resolve({ ok: false, reason: error.message }))
  })
}

function resolveBackendCommand() {
  if (app.isPackaged) {
    const name = process.platform === 'win32' ? 'driveguard-backend.exe' : 'driveguard-backend'
    const executable = path.join(process.resourcesPath, 'backend', name)
    if (!fs.existsSync(executable)) {
      throw new Error(`安装包缺少后端可执行文件：${executable}`)
    }
    return { command: executable, args: [], cwd: path.dirname(executable) }
  }

  const backendDir = path.join(repoRoot(), 'backend')
  const defaultPython =
    process.platform === 'win32'
      ? path.join(backendDir, '.venv', 'Scripts', 'python.exe')
      : path.join(backendDir, '.venv', 'bin', 'python')
  const python = process.env.DGB_DESKTOP_PYTHON || defaultPython
  if (!fs.existsSync(python)) {
    throw new Error(`开发模式找不到后端解释器：${python}（先运行 backend/run.sh 创建 .venv）`)
  }
  return {
    command: python,
    args: ['-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', String(backendPort)],
    cwd: backendDir,
  }
}

function startBackend(config) {
  const userData = app.getPath('userData')
  const dbPath = path.join(userData, 'driveguard.db').replace(/\\/g, '/')
  const env = {
    ...process.env,
    DGB_ADMIN_TOKEN: localToken,
    DGB_DATABASE_URL: `sqlite:///${dbPath}`,
    DGB_MEDIA_DIR: path.join(userData, 'media'),
    DGB_STATIC_DIR: bundledFrontend(),
    DGB_AUTO_CREATE_SCHEMA: 'true',
    DGB_SERVICE_VERSION: app.getVersion(),
    DGB_INFERENCE_HZ: String(config.inferenceHz ?? 5),
    DGB_INFERENCE_MODE: isDemo ? 'fake' : 'live',
    DGB_LIVE_SOURCE: 'camera',
    DGB_LIVE_VIEW: config.view || 'body',
    DGB_LIVE_IMAGE_SIZE: config.imageSize || '640x480',
    DGB_LIVE_ROI: config.driverRoi || '0.15,0.0,0.75,1.0',
    DGB_LIVE_STEP_MS: '40',
    DGB_CAMERA_DEVICE_INDEX: String(config.cameraIndex ?? 0),
    DGB_CAMERA_WIDTH: String(config.cameraWidth ?? 640),
    DGB_CAMERA_HEIGHT: String(config.cameraHeight ?? 480),
    DGB_CAMERA_JPEG_QUALITY: String(config.cameraQuality ?? 60),
    DGB_CAMERA_BACKEND: config.cameraBackend || '',
    DGB_PREVIEW_MAX_FPS: String(config.previewMaxFps ?? 10),
    DGB_DESKTOP_PORT: String(backendPort),
  }

  if (!isDemo) {
    const tokenPath = path.join(userData, 'model-service.token')
    fs.mkdirSync(path.dirname(tokenPath), { recursive: true })
    fs.writeFileSync(tokenPath, config.modelServiceToken.trim(), { mode: 0o600 })
    env.DGB_LIVE_URL = config.modelServiceUrl.trim()
    env.DGB_LIVE_TOKEN_FILE = tokenPath
  }

  const { command, args, cwd } = resolveBackendCommand()
  fs.mkdirSync(path.dirname(logFile()), { recursive: true })
  rotateLog()
  backendLog = fs.createWriteStream(logFile(), { flags: 'a' })
  appendLog(`\n=== ${new Date().toISOString()} start demo=${isDemo} ===\n`)

  backendProcess = spawn(command, args, {
    cwd,
    env,
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  backendError = null
  backendProcess.stdout.on('data', (chunk) => {
    appendLog(chunk.toString())
    process.stdout.write(chunk)
  })
  backendProcess.stderr.on('data', (chunk) => {
    appendLog(chunk.toString())
    process.stderr.write(chunk)
  })
  backendProcess.on('error', (error) => {
    backendError = error.message
    appendLog(`spawn error: ${error.message}\n`)
    if (!isQuitting && mainWindow) showCrash(`无法启动本地后端：${error.message}`)
  })
  backendProcess.on('exit', (code, signal) => {
    appendLog(`backend exit code=${code} signal=${signal}\n`)
    backendProcess = null
    if (!isQuitting && mainWindow) showCrash(`本地后端已退出（code=${code}）`)
  })
}

function stopBackend() {
  const child = backendProcess
  backendProcess = null
  if (child) {
    try {
      child.kill()
    } catch {
      // 已退出。
    }
  }
  if (backendLog) {
    try {
      backendLog.end()
    } catch {
      // 忽略关闭失败。
    }
    backendLog = null
  }
}

function showCrash(message) {
  if (isQuitting) return
  stopBackend()
  dialog.showErrorBox('DriveGuard AI 运行异常', `${message}\n\n日志目录：${path.dirname(logFile())}`)
  app.quit()
}

function createSplash() {
  splashWindow = new BrowserWindow({
    width: 420,
    height: 300,
    frame: false,
    resizable: false,
    show: true,
    backgroundColor: '#0b3b8c',
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true },
  })
  splashWindow.loadFile(path.join(__dirname, 'splash.html'))
  splashWindow.on('closed', () => {
    splashWindow = null
  })
}

function createMainWindow() {
  const isWindows = process.platform === 'win32'
  const options = {
    width: 1440,
    height: 920,
    minWidth: 1100,
    minHeight: 700,
    show: false,
    backgroundColor: '#f4f6fb',
    title: 'DriveGuard AI',
    autoHideMenuBar: true,
    titleBarStyle: isWindows ? 'hidden' : 'default',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      spellcheck: false,
      additionalArguments: [`--dgb-token=${localToken}`, `--dgb-version=${app.getVersion()}`],
    },
  }
  if (isWindows) {
    options.titleBarOverlay = { color: '#0b3b8c', symbolColor: '#ffffff', height: 40 }
  }
  if (!app.isPackaged) {
    options.icon = path.join(__dirname, '..', 'build', 'icon.png')
  }

  mainWindow = new BrowserWindow(options)
  mainWindow.loadURL(backendBase)

  mainWindow.once('ready-to-show', () => {
    if (mainWindow) mainWindow.show()
  })
  mainWindow.on('closed', () => {
    mainWindow = null
  })
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//i.test(url)) void shell.openExternal(url)
    return { action: 'deny' }
  })
  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith(backendBase)) {
      event.preventDefault()
      if (/^https?:\/\//i.test(url)) void shell.openExternal(url)
    }
  })
  mainWindow.webContents.on('did-fail-load', (_event, code, description) => {
    // -3 是导航被中止（退出/重定向），不是故障。
    if (isQuitting || code === -3) return
    showCrash(`界面加载失败（${code}）：${description}`)
  })
  mainWindow.webContents.on('before-input-event', (event, input) => {
    if (!app.isPackaged) return
    const key = (input.key || '').toLowerCase()
    const devtools = key === 'f12' || (input.control && input.shift && key === 'i')
    if (devtools) event.preventDefault()
  })
}

async function boot() {
  Menu.setApplicationMenu(null)
  const config = loadConfig()
  localToken = crypto.randomBytes(32).toString('hex')
  backendPort = await findFreePort()
  backendBase = `http://127.0.0.1:${backendPort}`

  createSplash()
  startBackend(config)
  await waitForHealth(backendPort)

  if (!isDemo) {
    const probe = await probeModelService(config)
    if (!probe.ok) {
      const choice = dialog.showMessageBoxSync({
        type: 'warning',
        title: '模型服务暂时不可达',
        message: '无法连接内置的模型服务',
        detail:
          `原因：${probe.reason}\n\n` +
          `服务地址：${config.modelServiceUrl}\n\n` +
          '应用仍可打开，但实时结果会保持等待并自动重试。请检查网络、服务地址和访问 Token。',
        buttons: ['仍然打开', '退出'],
        defaultId: 0,
        cancelId: 1,
        noLink: true,
      })
      if (choice === 1) {
        app.quit()
        return
      }
    }
  }

  if (!backendProcess) {
    throw new Error('本地后端在打开界面前退出，请查看日志')
  }

  createMainWindow()
  booting = false
  if (splashWindow) splashWindow.close()
}

const gotLock = app.requestSingleInstanceLock()
if (!gotLock) {
  app.quit()
} else {
  app.setAppUserModelId(APP_ID)

  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore()
      mainWindow.focus()
    }
  })

  app.on('before-quit', () => {
    isQuitting = true
    stopBackend()
  })

  app.on('window-all-closed', () => {
    if (!booting) app.quit()
  })

  app.on('activate', () => {
    if (mainWindow) mainWindow.focus()
  })

  ipcMain.handle('desktop:open-data-dir', async () => {
    await shell.openPath(app.getPath('userData'))
  })
  ipcMain.handle('desktop:open-logs', async () => {
    await shell.openPath(path.dirname(logFile()))
  })

  app.whenReady().then(boot).catch((error) => {
    const message = error && error.message ? error.message : String(error)
    try {
      appendLog(`FATAL ${message}\n`)
    } catch {
      // 忽略日志失败。
    }
    dialog.showErrorBox(
      'DriveGuard AI 启动失败',
      `${message}\n\n日志目录：${app.getPath('userData')}\\logs`,
    )
    app.quit()
  })
}
