export const isDebug = (search: string = location.search): boolean => new URLSearchParams(search).get("debug") === "1";
