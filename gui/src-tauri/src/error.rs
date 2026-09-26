//! 命令错误：统一转成一段可读的中文 / 英文消息，作为前端 `invoke` 的 reject 值。

use serde::{Serialize, Serializer};

/// 命令错误（序列化为字符串）。
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct CommandError(String);

impl CommandError {
    pub fn msg(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl From<anyhow::Error> for CommandError {
    fn from(e: anyhow::Error) -> Self {
        // `{:#}` 带上完整的原因链（如"打开文件失败：……：系统找不到指定的文件"）
        Self(format!("{e:#}"))
    }
}

impl From<tauri::Error> for CommandError {
    fn from(e: tauri::Error) -> Self {
        Self(e.to_string())
    }
}

impl Serialize for CommandError {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

/// 命令的返回类型。
pub type CmdResult<T> = Result<T, CommandError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_as_plain_string_with_cause_chain() {
        let e: CommandError = anyhow::anyhow!("inner").context("outer").into();
        assert_eq!(serde_json::to_string(&e).unwrap(), "\"outer: inner\"");
    }
}
