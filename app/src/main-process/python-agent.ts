// File: main-process/python-agent.ts
import { spawn } from 'child_process'

export function runCommitAgent(repoPath: string): Promise<{ title: string, description: string }> {
  return new Promise((resolve, reject) => {
    const process = spawn('python3', ['C:/Users/Dev_3/Desktop/Local Commit Generator/llm_git_agent.py', '--repo-path', repoPath, '--commit', 'false'])

    let output = ''
    process.stdout.on('data', data => output += data)
    process.stderr.on('data', data => console.error(`stderr: ${data}`))

    process.on('close', code => {
      const [title, ...desc] = output.trim().split('\n')
      resolve({
        title: title,
        description: desc.join('\n').trim()
      })
    })
  })
}
