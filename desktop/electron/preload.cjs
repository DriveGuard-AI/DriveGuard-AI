'use strict'

/**
 * 只暴露桌面版必需的最小能力；不向页面暴露 Node、文件系统或模型服务 Token。
 * 本地管理 Token 通过 BrowserWindow 的 additionalArguments 传入，用于调用本机后端。
 */

const { contextBridge, ipcRenderer } = require('electron')

function readArg(prefix) {
  const match = process.argv.find((value) => value.startsWith(prefix))
  return match ? match.slice(prefix.length) : ''
}

contextBridge.exposeInMainWorld('driveguard', {
  isDesktop: true,
  token: readArg('--dgb-token='),
  version: readArg('--dgb-version='),
  platform: process.platform,
  openDataDir: () => ipcRenderer.invoke('desktop:open-data-dir'),
  openLogs: () => ipcRenderer.invoke('desktop:open-logs'),
})
