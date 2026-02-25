/// <reference types="vite/client" />

interface ImportMetaEnv {
    readonly VITE_API_BASE_URL: string
    //другие переменные окружения по мере необходимости
}

interface ImportMeta {
    readonly env: ImportMetaEnv
}