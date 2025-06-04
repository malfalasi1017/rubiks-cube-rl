FROM node:22-alpine AS base
WORKDIR /app

# Install dependencies (including devDependencies for build)
FROM base AS deps
COPY package.json package-lock.json* ./
RUN npm ci && npm cache clean --force

# Build the Next.js app
FROM base AS builder
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# Production image with minimal footprint
FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy necessary files for standalone output
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/package.json ./package.json

USER nextjs

EXPOSE 3000
CMD ["node", "server.js"]