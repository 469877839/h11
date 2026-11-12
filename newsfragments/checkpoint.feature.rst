Connections can now export and restore versioned, pickle-free checkpoints
for offline replay of an immutable captured byte stream. Checkpoints cover the
unparsed receive buffer, EOF, roles, both state machines, message framing and
chunk progress, and pending protocol-switch proposals; corrupt or unsupported
checkpoints are rejected before use.
