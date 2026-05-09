import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  serverExternalPackages: ["@grpc/grpc-js", "@bufbuild/protobuf"],
};

export default nextConfig;
