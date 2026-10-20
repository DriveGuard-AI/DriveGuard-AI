/** Electron preload 暴露的最小桌面桥接接口（Web 构建中不存在）。 */

export interface DesktopBridge {
  readonly isDesktop: true
  /** 每次启动由主进程生成的本地后端管理 Token。 */
  readonly token: string
  readonly version: string
  readonly platform: string
  openDataDir: () => Promise<void>
  openLogs: () => Promise<void>
}

declare global {
  interface Window {
    driveguard?: DesktopBridge
  }
}

export {}
