export type ErrorType =
  | "invalid_request_error"
  | "authentication_error"
  | "permission_error"
  | "not_found_error"
  | "request_too_large"
  | "rate_limit_error"
  | "api_error"
  | "overloaded_error"
  | "timeout_error";

const STATUS: Record<ErrorType, number> = {
  invalid_request_error: 400,
  authentication_error: 401,
  permission_error: 403,
  not_found_error: 404,
  request_too_large: 413,
  rate_limit_error: 429,
  api_error: 500,
  overloaded_error: 529,
  timeout_error: 504,
};

export class ApiError extends Error {
  readonly type: ErrorType;
  readonly status: number;

  constructor(type: ErrorType, message: string) {
    super(message);
    this.name = "ApiError";
    this.type = type;
    this.status = STATUS[type];
  }

  toJSON(): { type: "error"; error: { type: ErrorType; message: string } } {
    return { type: "error", error: { type: this.type, message: this.message } };
  }
}

export function badRequest(message: string): ApiError {
  return new ApiError("invalid_request_error", message);
}

export function toApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err;
  const message = err instanceof Error ? err.message : String(err);
  return new ApiError("api_error", message);
}
