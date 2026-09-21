export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}
export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(`/api${path}`, {
    credentials: "same-origin",
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers,
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(
      typeof body.detail === "string"
        ? body.detail
        : "Не удалось выполнить запрос",
      response.status,
    );
  }
  return response.json();
}
export const pct = (n: number, digits = 1) =>
  new Intl.NumberFormat("ru-RU", {
    style: "percent",
    maximumFractionDigits: digits,
  }).format(n);
export const num = (n: number) => new Intl.NumberFormat("ru-RU").format(n);
export const date = (value: string, time = false) =>
  new Intl.DateTimeFormat("ru-RU", {
    day: "2-digit",
    month: "short",
    ...(time ? ({ hour: "2-digit", minute: "2-digit" } as const) : {}),
  }).format(new Date(value.slice(0, 19)));
export const clock = (value: string) => value.slice(11, 16);
export const kindNames = {
  fault: "Оборудование",
  fire: "Пожарный сигнал",
  flood: "Подтопление",
  access: "Доступ",
};
