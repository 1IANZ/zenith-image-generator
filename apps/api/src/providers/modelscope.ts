/**
 * ModelScope Provider Implementation
 */

import type { GenerateSuccessResponse } from '@z-image/shared'
import type { ImageProvider, ProviderGenerateRequest } from './types'

interface ModelScopeResponse {
  images?: Array<{ url?: string }>
}

export class ModelScopeProvider implements ImageProvider {
  readonly id = 'modelscope'
  readonly name = 'ModelScope'

  private readonly baseUrl = 'https://api-inference.modelscope.cn/v1'

  async generate(request: ProviderGenerateRequest): Promise<GenerateSuccessResponse> {
    if (!request.authToken) {
      throw new Error('API Token is required for ModelScope')
    }

    const body: Record<string, unknown> = {
      prompt: request.prompt,
      model: request.model || 'Tongyi-MAI/Z-Image-Turbo',
      size: `${request.width}x${request.height}`,
      seed: request.seed ?? Math.floor(Math.random() * 2147483647),
      steps: request.steps ?? 9,
    }

    if (request.guidanceScale !== undefined) {
      body.guidance = request.guidanceScale
    }

    const response = await fetch(`${this.baseUrl}/images/generations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${request.authToken.trim()}`,
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const text = await response.text()
      throw new Error(`ModelScope API error: ${response.status} - ${text}`)
    }

    const data = (await response.json()) as ModelScopeResponse
    const imageUrl = data.images?.[0]?.url

    if (!imageUrl) {
      throw new Error('No image returned from ModelScope')
    }

    return { url: imageUrl }
  }
}

export const modelscopeProvider = new ModelScopeProvider()
