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
/* Archive moments are naive Moscow time and are shown as written. Application
   timestamps carry an explicit UTC offset and are converted to Moscow time. */
export const date = (value: string, time = false) => {
  const zoned = /(Z|[+-]\d{2}:\d{2})$/.test(value);
  return new Intl.DateTimeFormat("ru-RU", {
    day: "2-digit",
    month: "short",
    ...(time ? ({ hour: "2-digit", minute: "2-digit" } as const) : {}),
    ...(zoned ? { timeZone: "Europe/Moscow" } : {}),
  }).format(new Date(zoned ? value : value.slice(0, 19)));
};
export const clock = (value: string) => value.slice(11, 16);
export const kindNames = {
  fault: "Оборудование",
  fire: "Пожарный сигнал",
  flood: "Подтопление",
  access: "Доступ",
};
